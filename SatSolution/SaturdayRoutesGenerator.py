import itertools
import pandas as pd


# Load data
duration = pd.read_csv(
    "../InputCSV/FoodstuffsDurations2026.csv",
    index_col=0
)

demand = pd.read_csv(
    "../InputCSV/PercentileDemand.csv",
    index_col=0
)


# Debugging counters
debug = True
final = 0
mid = 0
demand_debug = 0


# Route-generation parameters
capacity = 16
max_time = 6 * 60 * 60
pallet_unload_time = 18 * 60
max_route_size = 5
time_between = (1/3) * 60 * 60

demand_column = "Saturday P75 (rounded)"

all_stores = [
    store for store in duration.index
    if store != "Warehouse"
    and demand.loc[store, demand_column] > 0
]


def calculate_route_time(orders):
    global mid, final

    total_time = 0

    # Travel from the warehouse to the first store
    total_time += duration.loc["Warehouse", orders[0]]

    # Travel between stores and unload at each preceding store
    for i in range(len(orders) - 1):
        current_store = orders[i]
        next_store = orders[i + 1]

        total_time += duration.loc[current_store, next_store]

        total_time += (
            demand.loc[current_store, demand_column]
            * pallet_unload_time
        )

        if total_time > max_time:
            if debug:
                mid += 1
            return None

    # Unload at the final store
    final_store = orders[-1]

    total_time += (
        demand.loc[final_store, demand_column]
        * pallet_unload_time
    )

    # Return from the final store to the warehouse
    total_time += duration.loc[final_store, "Warehouse"]

    if total_time > max_time:
        if debug:
            final += 1
        return None

    return total_time


def calculate_neighbours(anchor):
    filtered_list = duration.loc[anchor] < time_between
    neighbours = filtered_list.index[filtered_list].tolist()

    neighbours = [
        store for store in neighbours
        if store in all_stores
    ]

    return neighbours

def brute_force_route(anchor):
    global demand_debug

    neighbours = calculate_neighbours(anchor)

    # This matches the weekday code and generates up to four-store routes
    route_length = min(
        len(neighbours),
        max_route_size - 1
    )

    output = []

    for r in range(1, route_length + 1):
        for combo in itertools.combinations(neighbours, r):

            # Ensure the route is generated around the current anchor
            if anchor not in combo:
                continue

            # Saturday P75 capacity requirement
            route_demand = sum(
                demand.loc[store, demand_column]
                for store in combo
            )

            if route_demand > capacity:
                demand_debug += 1
                continue

            # Test every possible ordering of these stores
            for perm in itertools.permutations(combo):
                route_time = calculate_route_time(perm)

                if route_time is not None:
                    output.append([
                        perm,
                        route_time,
                        route_demand
                    ])

    return output


def generate_all_routes(store_names):
    all_routes = {}

    for anchor in store_names:
        candidates = brute_force_route(anchor)

        for perm, route_time, total_demand in candidates:
            if perm not in all_routes:
                all_routes[perm] = [
                    route_time,
                    total_demand
                ]

    return [
        [perm, route_time, total_demand]
        for perm, [route_time, total_demand]
        in all_routes.items()
    ]


def save_routes_to_csv(
    routes,
    filename="InputCSV/saturday_routes.csv"
):
    data = []

    for perm, route_time, total_demand in routes:
        data.append({
            "Route": " -> ".join(perm),
            "Stops": list(perm),
            "Duration": route_time,
            "Number of Stops": len(perm),
            "Total Demand": total_demand
        })

    output = pd.DataFrame(data)
    output.to_csv(filename, index=False)

    return output


# Generate and save Saturday routes
all_routes = generate_all_routes(all_stores)

saturday_routes = save_routes_to_csv(
    all_routes,
    filename="../InputCSV/saturday_routes.csv"
)

print(f"Generated {len(saturday_routes)} Saturday routes.")

print(
    f"Failed due to time constraint: {mid + final}, "
    f"of which {final} failed during the "
    f"final-store and return stage. "
    f"{demand_debug} failed due to the demand constraint."
)
