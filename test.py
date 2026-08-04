import itertools

def route_drive_time(order, duration):
    t = duration["Warehouse"][order[0]]
    for a, b in zip(order[:-1], order[1:]):
        t += duration[a][b]
    t += duration[order[-1]]["Warehouse"]
    return t


def best_order_bruteforce(stores, duration):
    best_order, best_time = None, float("inf")
    for perm in itertools.permutations(stores):
        t = route_drive_time(perm, duration)
        if t < best_time:
            best_time, best_order = t, perm
    return list(best_order), best_time


def best_order_nearest_neighbour(stores, duration):
    remaining = set(stores)
    order = []
    current = "Warehouse"
    total_time = 0
    while remaining:
        nxt = min(remaining, key=lambda s: duration[current][s])
        total_time += duration[current][nxt]
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    total_time += duration[current]["Warehouse"]
    return order, total_time


def generate_feasible_routes(demand, duration, use_bruteforce=True):
    stores = list(demand.keys())
    routes = []
    route_id = 0

    for size in range(1, MAX_ROUTE_SIZE + 1):
        for subset in itertools.combinations(stores, size):

            pallets = sum(demand[s] for s in subset)
            if pallets > TRUCK_CAPACITY:
                continue

            if use_bruteforce and size <= 7:
                order, drive_time = best_order_bruteforce(subset, duration)
            else:
                order, drive_time = best_order_nearest_neighbour(subset, duration)

            unload_time = pallets * UNLOAD_SEC_PER_PALLET
            total_time = drive_time + unload_time

            if total_time <= UNLOAD_SEC_PER_PALLET:
                routes.append({
                    "route_id": route_id,
                    "stores": order,
                    "n_stores": size,
                    "pallets": pallets,
                    "drive_time_sec": drive_time,
                    "unload_time_sec": unload_time,
                    "total_time_sec": total_time,
                })
                route_id += 1

    return routes

TRUCK_CAPACITY = 16
SHIFT_LIMIT_SEC = 3.5 * 3600
UNLOAD_SEC_PER_PALLET = 18 * 60
MAX_ROUTE_SIZE = 4

demand = {
    "New World Albany": 5,
    "New World Birkenhead": 3,
    "Four Square Torbay": 2,
    "Pak 'n Save Albany": 6,
}

stores_all = ["Warehouse", "New World Albany", "New World Birkenhead", "Four Square Torbay", "Pak 'n Save Albany"]
duration = {a: {b: (0 if a == b else 900) for b in stores_all} for a in stores_all}

routes = generate_feasible_routes(demand, duration)
for r in routes:
    print(r)
