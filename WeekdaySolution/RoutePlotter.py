import pandas as pd
import matplotlib.pyplot as plt

WAREHOUSE = "Warehouse"


def parse_routes_dataframe(df, route_col="Route", truck_col="Truck"):
    # Convert each Route string into a list of individual supermarkets
    routes = [
        [s.strip() for s in route_str.split("->")]
        for route_str in df[route_col]
    ]

    # Get truck type for each route
    truck_types = (
        df[truck_col].tolist()
        if truck_col in df.columns
        else None
    )

    return routes, truck_types


def plot_routes(
    routes,
    locations_df,
    title="Truck Routes",
    figsize=(11, 11),
    save_path=None,
    truck_types=None
):

    # Use supermarket names as the index
    loc = locations_df.set_index("Supermarket")[["Long", "Lat"]]

    fig, ax = plt.subplots(figsize=figsize)

    # ---------------------------------------------------------
    # Plot every supermarket in the background
    # ---------------------------------------------------------
    ax.scatter(
        loc["Long"],
        loc["Lat"],
        color="lightgrey",
        s=25,
        zorder=1
    )

    # ---------------------------------------------------------
    # Plot warehouse
    # ---------------------------------------------------------
    wh = loc.loc[WAREHOUSE]

    ax.scatter(
        wh["Long"],
        wh["Lat"],
        color="black",
        marker="s",
        s=120,
        zorder=5,
        label="Warehouse"
    )

    # ---------------------------------------------------------
    # Give each route a different colour
    # ---------------------------------------------------------
    cmap = plt.get_cmap("tab20")

    for i, route in enumerate(routes):

        # Warehouse -> stores -> Warehouse
        stops = [WAREHOUSE] + route + [WAREHOUSE]

        # Get coordinates for every stop
        xs = [loc.loc[stop, "Long"] for stop in stops]
        ys = [loc.loc[stop, "Lat"] for stop in stops]

        # Different colour for every route
        colour = cmap(i % 20)

        # -----------------------------------------------------
        # Plot route
        # -----------------------------------------------------
        ax.plot(
            xs,
            ys,
            "-o",
            color=colour,
            linewidth=2,
            markersize=6,
            zorder=3
        )

        # -----------------------------------------------------
        # Highlight supermarket stops
        # -----------------------------------------------------
        ax.scatter(
            xs[1:-1],
            ys[1:-1],
            color=colour,
            s=60,
            zorder=4,
            edgecolor="black",
            linewidth=0.5
        )

    # ---------------------------------------------------------
    # Labels and formatting
    # ---------------------------------------------------------
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9
    )

    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    return fig, ax
if __name__ == "__main__":

    loc_df = pd.read_csv("../InputCSV/FoodstuffsLocations.csv")
    solved_df = pd.read_csv("../SatSolution/LpSolution_base_sat.csv")

    routes, truck_types = parse_routes_dataframe(solved_df)

    plot_routes(
        routes,
        loc_df,
        truck_types=truck_types,
        title="Saturday Schedule",
        save_path="../SatSolution/solved_schedule_sat.png"
    )

    print("Saved plot.")