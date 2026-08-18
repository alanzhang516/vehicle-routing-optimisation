import numpy as np
import pandas as pd

duration = pd.read_csv('../InputCSV/FoodstuffsDurations2026.csv', index_col=0)
demand = pd.read_csv('../InputCSV/PercentileDemand.csv', index_col=0)
pallet_unload_time = 18 * 60

def simulate_route_duration(route, pct_range=0.15):
    route_cleaned = ['Warehouse'] + route.split(" -> ") + ['Warehouse']
    total_duration = 0

    for i in range(len(route_cleaned) - 1):
        from_stop = route_cleaned[i]
        to_stop = route_cleaned[i + 1]

        multiplier = np.random.uniform(1 - pct_range, 1 + pct_range)
        total_duration += multiplier * duration.loc[from_stop, to_stop]

        if to_stop != 'Warehouse':
            total_duration += demand.loc[to_stop, "Weekday P75 (rounded)"] * pallet_unload_time

    return total_duration

