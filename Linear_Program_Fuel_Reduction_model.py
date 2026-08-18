import ast

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


# Input and output files
ROUTES_FILE = "WeekdaySolution/routes.csv"
STORES_FILE = "InputCSV/FoodstuffsLocations.csv"
LP_FILE = "FuelReduction.lp"
ROUTE_SOLUTION_FILE = "FuelReductionSolution.csv"
SKIPPED_STORES_FILE = "FuelReductionSkippedStores.csv"

# Model parameters
NUMBER_TRUCKS_AVAILABLE = 40
TRUCK_COST = 220
OVERTIME_COST = 310
HIRE_COST = 1400
HOURS_BEFORE_OVERTIME = 4
MAX_SKIPPED_STORES = 11

# Skipped-delivery penalties
PAKNSAVE_SKIP_PENALTY = 1500
OTHER_STORE_SKIP_PENALTY = 800


def prep_costs(file_routes):
    """Read the generated routes and calculate owned and Linfox costs."""
    routes = pd.read_csv(file_routes)

    # The CSV stores Stops as text, so convert each value back into a list.
    routes["Stops"] = routes["Stops"].apply(ast.literal_eval)

    # Route generation records duration in seconds.
    routes["Duration"] = routes["Duration"] / 3600

    # Owned trucks cost $220 per hour for the first four hours. Any overtime
    # is charged in complete hours at $310 per hour.
    routes["Cost"] = np.where(
        routes["Duration"] <= HOURS_BEFORE_OVERTIME,
        routes["Duration"] * TRUCK_COST,
        HOURS_BEFORE_OVERTIME * TRUCK_COST
        + np.ceil(routes["Duration"] - HOURS_BEFORE_OVERTIME) * OVERTIME_COST,
    )

    # Linfox charges $1,400 for each two-hour block or part thereof.
    routes["LinfoxCost"] = (
        HIRE_COST * np.ceil(routes["Duration"] / 2)
    )

    return routes.reset_index(drop=True)


def prep_patterns(file_stores, routes):
    """Create the store-route incidence patterns used by the model."""
    store_data = pd.read_csv(file_stores)

    # The depot appears as "Warehouse" in the Supermarket column, but it is
    # only the start and end point of each route and must not be covered.
    supermarket_names = store_data["Supermarket"].dropna().astype(str)
    store_names = (
        supermarket_names[
            supermarket_names.str.strip().str.casefold() != "warehouse"
        ]
        .drop_duplicates()
        .tolist()
    )

    route_stops = [set(stops) for stops in routes["Stops"]]

    patterns = {
        store: [int(store in stops) for stops in route_stops]
        for store in store_names
    }

    missing_stores = [
        store
        for store, route_visits in patterns.items()
        if not any(route_visits)
    ]

    if missing_stores:
        raise ValueError(
            f"No feasible route generated for: {', '.join(missing_stores)}"
        )

    return patterns


def get_skip_penalties(store_names):
    """Charge $1,500 for skipping Pak'nSave and $800 for any other store."""
    penalties = {}

    for store in store_names:
        normalised_name = store.strip().casefold()
        is_paknsave = normalised_name.startswith("pak")

        penalties[store] = (
            PAKNSAVE_SKIP_PENALTY
            if is_paknsave
            else OTHER_STORE_SKIP_PENALTY
        )

    return penalties


def save_solution(
    routes,
    route_names,
    route_owned,
    route_linfox,
    store_names,
    store_skipped,
    skip_penalties,
):
    """Save the selected routes and skipped stores to separate CSV files."""
    selected_routes = []

    for route_id in route_names:
        truck_type = None
        route_cost = None

        if value(route_owned[route_id]) > 0.5:
            truck_type = "Owned"
            route_cost = routes.loc[route_id, "Cost"]
        elif value(route_linfox[route_id]) > 0.5:
            truck_type = "Linfox"
            route_cost = routes.loc[route_id, "LinfoxCost"]

        if truck_type is not None:
            selected_routes.append({
                "Route ID": route_id,
                "Route": routes.loc[route_id, "Route"],
                "Stops": routes.loc[route_id, "Stops"],
                "Duration (hours)": routes.loc[route_id, "Duration"],
                "Truck": truck_type,
                "Route Cost": route_cost,
            })

    skipped_stores = [
        {
            "Store": store,
            "Penalty": skip_penalties[store],
        }
        for store in store_names
        if value(store_skipped[store]) > 0.5
    ]

    selected_routes_df = pd.DataFrame(selected_routes)
    skipped_stores_df = pd.DataFrame(
        skipped_stores,
        columns=["Store", "Penalty"],
    )

    selected_routes_df.to_csv(ROUTE_SOLUTION_FILE, index=False)
    skipped_stores_df.to_csv(SKIPPED_STORES_FILE, index=False)

    return selected_routes_df, skipped_stores_df


def create_fuel_reduction_model(routes, patterns, skip_penalties):
    """Create and solve the routing model with optional missed deliveries."""
    route_names = routes.index.tolist()
    store_names = list(patterns.keys())
    store_ids = range(len(store_names))

    route_owned = LpVariable.dicts(
        "Route_Owned",
        route_names,
        cat=LpBinary,
    )
    route_linfox = LpVariable.dicts(
        "Route_Linfox",
        route_names,
        cat=LpBinary,
    )
    skipped_by_id = LpVariable.dicts(
        "Store_Skipped",
        store_ids,
        cat=LpBinary,
    )
    store_skipped = {
        store: skipped_by_id[store_id]
        for store_id, store in enumerate(store_names)
    }

    model = LpProblem("Fuel_Reduction_Model", LpMinimize)

    # Minimise routing costs plus the penalties for missed deliveries.
    model += (
        lpSum(
            route_owned[route_id] * routes.loc[route_id, "Cost"]
            + route_linfox[route_id] * routes.loc[route_id, "LinfoxCost"]
            for route_id in route_names
        )
        + lpSum(
            store_skipped[store] * skip_penalties[store]
            for store in store_names
        ),
        "Total Cost",
    )

    # Every store must either be visited exactly once or be skipped.
    for store_id, store in enumerate(store_names):
        model += (
            lpSum(
                (route_owned[route_id] + route_linfox[route_id])
                * patterns[store][route_id]
                for route_id in route_names
            )
            + store_skipped[store]
            == 1,
            f"Store_Coverage_{store_id}",
        )

    # At most 40 routes may use owned Foodstuffs trucks.
    model += (
        lpSum(route_owned[route_id] for route_id in route_names)
        <= NUMBER_TRUCKS_AVAILABLE,
        "Owned_Truck_Limit",
    )

    # At most 11 stores may have their delivery deferred.
    model += (
        lpSum(store_skipped[store] for store in store_names)
        <= MAX_SKIPPED_STORES,
        "Skipped_Store_Limit",
    )

    # A route cannot simultaneously use an owned and a Linfox truck.
    for route_id in route_names:
        model += (
            route_owned[route_id] + route_linfox[route_id] <= 1,
            f"One_Truck_Per_Route_{route_id}",
        )

    model.writeLP(LP_FILE)
    model.solve(PULP_CBC_CMD(msg=True))

    status = LpStatus[model.status]
    print(f"Status: {status}")

    if status != "Optimal":
        raise RuntimeError(f"The model did not find an optimal solution: {status}")

    selected_routes, skipped_stores = save_solution(
        routes,
        route_names,
        route_owned,
        route_linfox,
        store_names,
        store_skipped,
        skip_penalties,
    )

    routing_cost = selected_routes["Route Cost"].sum()
    penalty_cost = skipped_stores["Penalty"].sum()
    owned_routes = selected_routes[selected_routes["Truck"] == "Owned"]
    linfox_routes = selected_routes[selected_routes["Truck"] == "Linfox"]

    print(f"Total objective cost: ${value(model.objective):,.2f}")
    print(f"Routing cost: ${routing_cost:,.2f}")
    print(f"Skipped-delivery penalties: ${penalty_cost:,.2f}")
    print(f"Owned routes selected: {len(owned_routes)}")
    print(f"Linfox routes selected: {len(linfox_routes)}")
    print(f"Stores skipped: {len(skipped_stores)}")

    if not owned_routes.empty:
        average_owned_duration = owned_routes["Duration (hours)"].mean()
        print(
            "Average owned-route duration: "
            f"{average_owned_duration:.2f} hours"
        )

    return model


if __name__ == "__main__":
    route_data = prep_costs(ROUTES_FILE)
    store_patterns = prep_patterns(STORES_FILE, route_data)
    penalties = get_skip_penalties(store_patterns.keys())

    create_fuel_reduction_model(
        route_data,
        store_patterns,
        penalties,
    )
