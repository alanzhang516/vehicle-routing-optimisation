import itertools
import pandas as pd

duration = pd.read_csv('../InputCSV/FoodstuffsDurations2026.csv', index_col=0)
demand = pd.read_csv('../InputCSV/PercentileDemand.csv', index_col=0)

debug = True
final = 0
mid = 0
demand_debug = 0

capacity = 16
max_time = 6 * 60 * 60
pallet_unload_time = 18 * 60
max_route_size = 5
time_between = 2.5 * 60 * 60
all_stores = duration.index.tolist()
all_stores.remove("Warehouse")

# print(duration.loc['Warehouse', 'New World Albany'])
# print(len(all_stores))

def calculate_route_time(orders):
    global mid, final
    total_time = 0
    total_time += duration.loc["Warehouse", orders[0]]

    for i in range(len(orders)-1):
        total_time += duration.loc[orders[i], orders[i+1]]
        total_time += demand.loc[orders[i], "Weekday P75 (rounded)"] * pallet_unload_time

        if total_time > max_time:
            if debug:
                mid += 1
            return None

    total_time += duration.loc[orders[-1], "Warehouse"]
    total_time += demand.loc[orders[-1], "Weekday P75 (rounded)"] * pallet_unload_time

    if total_time > max_time:
        if debug:
            final += 1
        return None

    return total_time

def calculate_neighbours(anchor):
    filtered_list = duration.loc[anchor] < time_between
    neighbours = filtered_list.index[filtered_list].tolist()

    return neighbours

def brute_force_route(anchor):
    global demand_debug
    neighbours = calculate_neighbours(anchor)
    if 'Warehouse' in neighbours:
        neighbours.remove('Warehouse')

    route_length = min(len(neighbours), max_route_size - 1)
    output = []

    for r in range(1, route_length + 1):
        for combo in itertools.combinations(neighbours, r):
            if anchor not in combo:
                continue

            route_demand = sum(demand.loc[store, "Weekday P75 (rounded)"] for store in combo)
            if route_demand > capacity:
                demand_debug += 1
                continue

            for perm in itertools.permutations(combo):
                t = calculate_route_time(perm)
                if t is not None:
                    output.append([perm, t, route_demand])
    return output

def generate_all_routes(store_names):
    all_routes = {}

    for anchor in store_names:
        candidates = brute_force_route(anchor)
        for perm, time, total_demand in candidates:
            if perm not in all_routes:
                all_routes[perm] = [time, total_demand]

    return [[perm, time, total_demand] for perm, [time, total_demand] in all_routes.items()]

def save_routes_to_csv(routes, filename="routes.csv"):
    data = []
    for perm, time, total_demand in routes:
        data.append({
            "Route": " -> ".join(perm),
            "Stops": list(perm),
            "Duration": time,
            "Number of Stops": len(perm),
            "Total Demand": total_demand
        })

    output = pd.DataFrame(data)
    output.to_csv(filename, index=False)
    return output

all_routes = generate_all_routes(all_stores)
save_routes_to_csv(all_routes)
print(f"Failed due to Time constraint: {mid + final} out of which {final} were due to the return trip from the final trip to the warehouse. {demand_debug} failed due to the demand constraint")
