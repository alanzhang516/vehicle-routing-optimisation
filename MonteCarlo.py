import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

UNLOAD_SEC  = 18 * 60
TRUCK_CAP   = 16
DEPOT       = 'Warehouse'      # must match the label used in the durations CSV
FLEET_SIZE  = 20               # routes 0-19 depart 8am, the rest depart 2pm
SHIFT_HOURS = (8, 14)

plot_data = pd.DataFrame(columns=['Weekday base', 'Weekday shedding', 'Saturday base', 'Saturday shedding'])
durations = pd.read_csv('InputCSV/FoodstuffsDurations2026.csv', index_col=0)

# traffic adjustment factors: mean and SD per hour, from the TomTom Auckland
# congestion levels. Converted once here into lognormal (mu, sigma).
_tf = {
    'weekday':  pd.read_csv('InputCSV/traffic_factor_weekday.csv',  index_col='Hour'),
    'saturday': pd.read_csv('InputCSV/traffic_factor_saturday.csv', index_col='Hour'),
}
_lognorm = {}
for _day, _t in _tf.items():
    for _h, _row in _t.iterrows():
        _m, _s = _row['Mean'], _row['SD']
        _sig = np.sqrt(np.log(1 + (_s / _m) ** 2))
        _lognorm[(_day, _h)] = (np.log(_m) - _sig ** 2 / 2, _sig)

# historical demand: stores as rows, one column per day
_demand = pd.read_csv('InputCSV/FoodstuffsDemand2026.csv', index_col=0)
# clean known outliers
_demand.at["Four Square Ellerslie", "3/06/2026"] = np.nan
_demand.at["Pak 'n Save Henderson", "10/06/2026"] = np.nan

stores  = list(_demand.index)

# split the 56 days into weekdays and Saturdays.
# dates are NZ format (day first), so the format is given explicitly - left to
# guess, pandas reads 05/06/2026 as 5 June rather than 6 May and the split is
# silently wrong. Also remove Monday public holiday
_dates = pd.to_datetime(_demand.columns, format='%d/%m/%Y')
_dow = _dates.dayofweek
public_holiday = pd.Timestamp("2026-06-01")
weekday_mask = (_dow <= 4) & (_dates != public_holiday)
saturday_mask = (_dow == 5)
pools = {
    'weekday': _demand.loc[:, weekday_mask].T,
    'saturday': _demand.loc[:, saturday_mask].T,
}

def sample_demand(day_type='weekday'):
    """One simulated day -> {store: pallets} for all 55 stores.

    Picks one historical day of the right type at random (with replacement)
    and takes every store's pallet count from that day."""
    pool = pools[day_type]
    # randomly choose one historical day
    sampled_day = pool.iloc[rng.integers(len(pool))].copy()
    # If that day contains a removed outlier, replace only that store's value
    for store in sampled_day.index:
        if pd.isna(sampled_day[store]):
            valid_values = pool[store].dropna()
            sampled_day[store] = rng.choice(valid_values)
    return sampled_day.to_dict()


def route_travel_time(route, day_type='weekday', hour=8):
    """Driving time for one route in seconds, Warehouse -> stops -> Warehouse.
    One traffic multiplier per trip, drawn from the lognormal fitted to the
    TomTom mean and SD for that day type and departure hour."""
    path = [DEPOT] + route.split(' -> ') + [DEPOT]
    mu, sigma = _lognorm[(day_type, hour)]
    mult = rng.lognormal(mu, sigma)

    return mult * sum(
        durations.loc[path[i], path[i+1]] for i in range(len(path) - 1)
    )


def monte_carlo(routes, shed_stores, day_type='weekday', n=5000):
    """Run n simulated days. Returns a DataFrame of per-day costs."""
    fixed = sum(1500 if "Pak 'n Save" in s else 800 for s in shed_stores)

    results = []
    for _ in range(n):
        demand = sample_demand(day_type)
        total, leases, over_six = fixed, 0, 0

        for i, route in enumerate(routes):
            hour    = SHIFT_HOURS[0] if i < FLEET_SIZE else SHIFT_HOURS[1]
            pallets = sum(demand[s] for s in route.split(' -> '))
            travel  = route_travel_time(route, day_type, hour)

            # owned truck takes up to 16 pallets
            carried = min(pallets, TRUCK_CAP)
            hours   = (travel + carried * UNLOAD_SEC) / 3600
            if hours > 6:
                over_six += 1
            total  += 220 * hours if hours <= 4 else 220 * 4 + 310 * math.ceil(hours - 4)

            # anything left over goes to a wet-leased truck
            left = pallets - carried
            while left > 0:
                carried = min(left, TRUCK_CAP)
                hours   = (route_travel_time(route, day_type, hour)
                           + carried * UNLOAD_SEC) / 3600
                if hours > 6:
                    over_six += 1
                total  += 1400 * math.ceil(hours / 2)
                leases += 1
                left   -= carried

        results.append((total, leases, over_six))

    return pd.DataFrame(results, columns=['cost', 'leases', 'over_six'])


SOLUTIONS = {
    'Weekday base':     ('WeekdaySolution/LpSolution_base.csv',     'weekday'),
    'Weekday shedding': ('WeekdaySolution/LpSolution_shedding.csv', 'weekday'),
    'Saturday base':    ('SatSolution/LpSolution_base_sat.csv',     'saturday'),
    'Saturday shedding':('SatSolution/LpSolution_shedding_sat.csv', 'saturday'),
}


def run_all(n=5000):
    """Run every scenario and return a summary table."""
    rows = []
    for label, (path, day_type) in SOLUTIONS.items():
        sol    = pd.read_csv(path)
        routes = list(sol['Route'])
        served = {s for r in routes for s in r.split(' -> ')}
        if day_type == 'saturday':
            active_stores = {
                store for store in stores
                if _demand.loc[store, saturday_mask].max() > 0
            }
        else:
            active_stores = set(stores)

        shed = active_stores - served

        df = monte_carlo(routes, shed, day_type, n=n)
        c  = df.cost
        se = c.std(ddof=1) / math.sqrt(n)

        plot_data[label] = c

        rows.append({
            'Scenario':      label,
            'Mean':          c.mean(),
            'CI low':        c.mean() - 1.96 * se,
            'CI high':       c.mean() + 1.96 * se,
            'Std dev':       c.std(ddof=1),
            '5th pct':       c.quantile(0.05),
            '95th pct':      c.quantile(0.95),
            'Worst':         c.max(),
            'Stores shed':   len(shed),
            'P(lease)':      (df.leases > 0).mean(),
            'Mean leases':   df.leases.mean(),
            'P(over 6h)':    (df.over_six > 0).mean(),
            'Mean no of over 6h': df.over_six.mean()
        })

    return pd.DataFrame(rows).set_index('Scenario')


def show(table):
    """Print the summary table with dollar and percent formatting."""
    money = ['Mean', 'CI low', 'CI high', 'Std dev', '5th pct', '95th pct', 'Worst']
    out = table.copy()
    for col in money:
        out[col] = out[col].map('${:,.0f}'.format)
    out['P(lease)']    = out['P(lease)'].map('{:.1%}'.format)
    out['Mean leases'] = out['Mean leases'].map('{:.2f}'.format)
    print(out.to_string())
    plt.boxplot(plot_data.values, tick_labels=plot_data.columns)
    plt.title('Box Plot of Cost over different scenarios')
    plt.ylabel('Cost (NZD)')
    plt.show()


if __name__ == '__main__':
    print(f'{len(stores)} stores, {len(pools["weekday"])} weekdays, '
          f'{len(pools["saturday"])} Saturdays\n')

    table = run_all(n=2000)
    show(table)

    table.to_csv('simulation_results.csv')
    print('\nwritten to simulation_results.csv')