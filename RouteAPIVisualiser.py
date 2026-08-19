import pandas as pd
import folium
import webbrowser
import openrouteservice as ors

locations = pd.read_csv("InputCSV/FoodstuffsLocations.csv")

coords = locations[['Long', 'Lat']]
coords = coords.to_numpy().tolist()

client = ors.Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjVkZjhlYTIzYTdiZjQ0ZDZhMjY3ODBkOWIyNmE2MDk4IiwiaCI6Im11cm11cjY0In0=")

routes_data = pd.read_csv("SatSolution/LpSolution_shedding_sat.csv")
routes_string = routes_data.Route

m = folium.Map(location=list(reversed(coords[2])), zoom_start=10)

for i in range(len(locations)):
    if locations.Type[i] == "Four Square":
        iconCol = "green"
    elif locations.Type[i] == "New World":
        iconCol = "red"
    elif locations.Type[i] == "Pak 'n Save":
        iconCol = "orange"
    elif locations.Type[i] == "Warehouse":
        iconCol = "black"
    folium.Marker(list(reversed(coords[i])), popup=locations.Supermarket[i], icon=folium.Icon(color=iconCol)).add_to(m)

colors = ['blue', 'purple', 'darkgreen', 'cadetblue', 'darkred']
route_summary_rows = []
leg_rows = []

warehouse_idx = (locations['Supermarket'] == 'Warehouse').idxmax()
warehouse_coord = coords[warehouse_idx]

for i, route in enumerate(routes_string):
    cleaned_route = [s.strip() for s in route.split(" -> ")]

    # Build the full stop sequence: Warehouse -> stops -> Warehouse
    full_route = ['Warehouse'] + cleaned_route + ['Warehouse']

    route_coordinates = []
    bad_stops = []

    for stop in full_route:
        match = locations['Supermarket'] == stop
        if not match.any():
            bad_stops.append(stop)
            continue
        index = match.idxmax()
        lon, lat = coords[index]
        if pd.isna(lon) or pd.isna(lat):
            bad_stops.append(f"{stop} (missing coords)")
            continue
        route_coordinates.append([lon, lat])

    if bad_stops:
        print(f"Route {i} skipped — couldn't resolve: {bad_stops}")
        continue

    if len(route_coordinates) < 2:
        print(f"Route {i} skipped — only {len(route_coordinates)} valid stop(s): {full_route}")
        continue

    try:
        route_geo = client.directions(
            coordinates=route_coordinates,
            profile='driving-hgv',
            format='geojson',
            validate=False
        )
    except ors.exceptions.ApiError as e:
        print(f"Route {i} failed at ORS: {full_route}")
        print(f"  error: {e}")
        continue

    folium.PolyLine(
        locations=[list(reversed(coord)) for coord in route_geo['features'][0]['geometry']['coordinates']],
        color=colors[i % len(colors)],
        weight=4,
        popup=f"Route {i+1}").add_to(m)

    props = route_geo['features'][0]['properties']
    summary = props['summary']

    route_summary_rows.append({
        'route_id': i,
        'route': route,
        'num_stops': len(full_route),
        'total_distance_m': summary['distance'],
        'total_duration_s': summary['duration'],
    })

    for leg_idx, segment in enumerate(props['segments']):
        leg_rows.append({
            'route_id': i,
            'leg_idx': leg_idx,
            'from_stop': full_route[leg_idx],
            'to_stop': full_route[leg_idx + 1],
            'distance_m': segment['distance'],
            'duration_s': segment['duration'],
        })

pd.DataFrame(route_summary_rows).to_csv('route_durations_summary.csv', index=False)

pd.DataFrame(leg_rows).to_csv('route_durations_legs.csv', index=False)

map_file = "Saturday_shedding_map.html"
m.save(map_file)
webbrowser.open(map_file)