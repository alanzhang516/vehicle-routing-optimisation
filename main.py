import itertools
import pandas as pd
import numpy as np

duration = pd.read_csv('Data/FoodstuffsDurations2026.csv', index_col=0)

capacity = 16
max_time = 3.5 * 60 * 60
pallet_unload_time = 18 * 60
max_route_size = 4
time_between = 0.5 * 60 * 60
all_stores = duration.index.tolist()
all_stores.remove("Warehouse")

# print(duration.loc['Warehouse', 'New World Albany'])
# print(len(all_stores))

def calculate_route_time(orders):
    total_time = 0
    total_time += duration.loc["Warehouse", orders[0]]

    for i in range(len(orders)-1):
        total_time += duration.loc[orders[i], orders[i+1]]

        if total_time > max_time:
            return None

    total_time += duration.loc[orders[-1], "Warehouse"]
    return total_time

def calculate_neighbours(anchor):
    filtered_list = duration.loc[anchor] < time_between
    neighbours = filtered_list.index[filtered_list].tolist()

    return neighbours

def brute_force_route(anchor):
    neighbours = calculate_neighbours(anchor)
    if 'Warehouse' in neighbours:
        neighbours.remove('Warehouse')

    route_length = min(len(neighbours), max_route_size - 1)
    output = []

    for r in range(1, route_length + 1):

        for perm in itertools.permutations(neighbours, r):
            if anchor not in perm:
                continue

            t = calculate_route_time(perm)

            if t is not None:
                output.append([perm, t])

    return output

def generate_all_routes(store_names):
    all_routes = {}

    for anchor in store_names:
        candidates = brute_force_route(anchor)
        for perm, t in candidates:
            if perm not in all_routes:
                all_routes[perm] = t

    return [[perm, t] for perm, t in all_routes.items()]

def save_routes_to_csv(routes, filename="routes.csv"):
    data = []
    for perm, t in routes:
        data.append({
            "Route": " -> ".join(perm),
            "Stops": list(perm),
            "Duration": t,
            "Number of Stops": len(perm)
        })

    output = pd.DataFrame(data)
    output.to_csv(filename, index=False)
    return output

all_routes = generate_all_routes(all_stores)
save_routes_to_csv(all_routes)
