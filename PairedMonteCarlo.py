import math
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt

import MonteCarlo as mc

def evaluate_solution(routes, shed_stores, demand, traffic, day_type):
    """ Evaluates one fixed routing solution for a single simulated day using
         pre-sampled demand and traffic conditions. This allows base and shedding
        scenarios to be compared under exactly the same random conditions."""
    fixed = sum(
        1500 if "Pak 'n Save" in s else 800
        for s in shed_stores
    )

    if day_type == 'saturday':
        more_congested = mc.SHIFT_HOURS[1]
        less_congested = mc.SHIFT_HOURS[0]
    else:
        more_congested = mc.SHIFT_HOURS[0]
        less_congested = mc.SHIFT_HOURS[1]

    cutoff = max(0, len(routes) - mc.FLEET_SIZE)

    total = fixed
    leases = 0
    over_six = 0
    for i, route in enumerate(routes):
        if i < cutoff:
            hour = more_congested
        else:
            hour = less_congested

        pallets = sum(
            demand[s] for s in route.split(' -> ')
        )

        travel = mc.route_travel_time(
            route,
            traffic[hour]
        )

        carried = min(pallets, mc.TRUCK_CAP)

        hours = (
                        travel + carried * mc.UNLOAD_SEC
                ) / 3600
        if hours > 6:
            over_six += 1

        total += (
            220 * hours
            if hours <= 4
            else 220 * 4 + 310 * math.ceil(hours - 4)
        )
        #any extra demand is delivered by leased trucks
        left = pallets - carried

        while left > 0:
            carried = min(left, mc.TRUCK_CAP)

            hours = (
                            mc.route_travel_time(route, traffic[hour])
                            + carried * mc.UNLOAD_SEC
                    ) / 3600

            if hours > 6:
                over_six += 1

            total += 1400 * math.ceil(hours / 2)
            leases += 1
            left -= carried
    return total, leases, over_six

def paired_monte_carlo(base_routes, base_shed,
                       shedding_routes, shedding_shed,
                       day_type='weekday', n=5000):
    """Sample demand and traffic once per day (with traffic one for each shift) then
    find results for each scenario"""
    results = []
    for _ in range(n):
        demand = mc.sample_demand(day_type)
        traffic = mc.sample_traffic(day_type)
        base_result = evaluate_solution(
        base_routes,
        base_shed,
        demand,
        traffic,
        day_type
        )

        shedding_result = evaluate_solution(
        shedding_routes,
        shedding_shed,
        demand,
        traffic,
        day_type
        )
        base_cost, base_leases, base_over_six = base_result
        shed_cost, shed_leases, shed_over_six = shedding_result
        results.append((
        base_cost,
        shed_cost,
        base_cost - shed_cost,
        base_leases,
        shed_leases,
        base_over_six,
        shed_over_six
        ))
    return pd.DataFrame(
        results,
        columns=[
            'base_cost',
            'shed_cost',
            'cost_difference',
            'base_leases',
            'shed_leases',
            'base_over_six',
            'shed_over_six'
        ]
    )
def load_pair(base_path, shedding_path, day_type):
    """Load and sort the base and shedding routing solutions for one day type."""
    base_sol = pd.read_csv(base_path)
    shedding_sol = pd.read_csv(shedding_path)

    base_routes = sorted(
        base_sol['Route'],
        key=mc.route_base_duration
    )

    shedding_routes = sorted(
        shedding_sol['Route'],
        key=mc.route_base_duration
    )
    base_served = {
        s for r in base_routes
        for s in r.split(' -> ')
    }

    shedding_served = {
        s for r in shedding_routes
        for s in r.split(' -> ')
    }
    if day_type == 'saturday':
        active_stores = {
            store for store in mc.stores
            if mc._demand.loc[store, mc.saturday_mask].max() > 0
        }
    else:
        active_stores = set(mc.stores)
    base_shed = active_stores - base_served
    shedding_shed = active_stores - shedding_served
    return base_routes, base_shed, shedding_routes, shedding_shed


def run_pair(base_path, shedding_path, day_type, n=5000):
    """Run a paired Monte Carlo comparison for base vs shedding."""

    base_routes, base_shed, shedding_routes, shedding_shed = load_pair(
        base_path,
        shedding_path,
        day_type
    )

    df = paired_monte_carlo(
        base_routes,
        base_shed,
        shedding_routes,
        shedding_shed,
        day_type=day_type,
        n=n
    )

    return df
def summarise_pair(df):
    """Summarise the paired cost differences between base and shedding."""

    diff = df['cost_difference']
    n = len(diff)

    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se_diff = sd_diff / math.sqrt(n)
    t_crit = t.ppf(0.975, df=n - 1)

    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff
    base_p_lease = (df['base_leases'] > 0).mean()
    shed_p_lease = (df['shed_leases'] > 0).mean()

    base_mean_leases = df['base_leases'].mean()
    shed_mean_leases = df['shed_leases'].mean()

    base_p_over_six = (df['base_over_six'] > 0).mean()
    shed_p_over_six = (df['shed_over_six'] > 0).mean()

    base_mean_over_six = df['base_over_six'].mean()
    shed_mean_over_six = df['shed_over_six'].mean()
    return {
        'Mean difference': mean_diff,
        'CI low': ci_low,
        'CI high': ci_high,
        'Std dev difference': sd_diff,
        'Base P(lease)': base_p_lease,
        'Shed P(lease)': shed_p_lease,
        'Base mean leases': base_mean_leases,
        'Shed mean leases': shed_mean_leases,
        'Base P(over 6h)': base_p_over_six,
        'Shed P(over 6h)': shed_p_over_six,
        'Base mean over 6h': base_mean_over_six,
        'Shed mean over 6h': shed_mean_over_six
    }
def show_paired_plot(df, title):
    """Show the distribution of paired cost differences."""

    diff = df['cost_difference']

    plt.hist(diff, bins=40)

    plt.axvline(
        diff.mean(),
        linestyle='--',
        linewidth=1,
        label=f'Mean difference = ${diff.mean():,.0f}'
    )

    plt.axvline(
        0,
        linestyle=':',
        linewidth=1,
        label='No difference'
    )

    plt.title(title)
    plt.xlabel('Base cost - shedding cost (NZD)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    weekday_df = run_pair(
        'WeekdaySolution/LpSolution_base.csv',
        'WeekdaySolution/LpSolution_shedding.csv',
        'weekday',
        n=5000
    )

    saturday_df = run_pair(
        'SatSolution/LpSolution_base_sat.csv',
        'SatSolution/LpSolution_shedding_sat.csv',
        'saturday',
        n=5000
    )
    summary = pd.DataFrame(
        [
            summarise_pair(weekday_df),
            summarise_pair(saturday_df)
        ],
        index=[
            'Weekday base vs shedding',
            'Saturday base vs shedding'
        ]
    )
    print(summary.to_string())
    show_paired_plot(
        weekday_df,
        'Weekday Paired Cost Differences'
    )

    summary.to_csv('paired_simulation_results.csv')
    print('\nwritten to paired_simulation_results.csv')

