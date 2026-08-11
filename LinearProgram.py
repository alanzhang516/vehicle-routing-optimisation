import numpy as np
import pandas as pd
from pulp import *

#######################################

# Variables for easily changing parameters in constraints etc.
numberTrucksAvailable = 40 # Number of trucks able to send out in a day without leasing more
truckCost = 220 # Cost of operating an owned truck ($ / hr)
overtimeCost = 310 # Cost of overtime for owned trucks ($ / hr)
hireCost = 1400 # Cost of hiring a truck from Linfox for 2 hours
hrsB4Overtime = 4.0 # Time each owned truck is available for without overtime



def prepCosts(fileRoutes):
    '''
    Calculates the cost of each route using an owned truck and a leased truck

    :param fileRoutes: file name for file that contains routes

    :returns: a dataframe containing the costs of each route

    NOTE:
        Overtime costing has been calculated assuming that it must be paid in full hours (i.e. 30 mins of overtime costs 1 hour)
         - to change remove the np.ceil function from cost calculation

        Leased costs are also calculated assuming that we pay in 2 hour blocks
         - to change remove the np.ceil function from cost calculation
    '''


    # Read generated routes
    routes = pd.read_csv(fileRoutes)

    # Convert from seconds to hours
    routes['Duration'] =routes.Duration / 3600

    # Add column Cost, for the cost of running the route with an owned truck
    routes['Cost'] = np.where(routes.Duration <= hrsB4Overtime, routes.Duration * truckCost, np.ceil(routes.Duration - hrsB4Overtime) * overtimeCost + hrsB4Overtime * truckCost)

    # Add column LinfoxCost, for the cost of running the route with a leased truck
    routes['LinfoxCost'] = hireCost * np.ceil(routes.Duration / 2)

    return routes

def prepPatterns(fileStores, routes):
    '''
    Generate the 'patterns' for the linear program

    :param fileStores: file name for file that contains stores
    :param routes: dataframe containing the routes

    :returns: a dictionary containing the patterns, and a dictionary containing the demands (Number of trucks needed to visit)
    '''
    # Read list of stores
    df = pd.read_csv(fileStores)
    #remove Warehouse from store delivery constraints
    df = df[df["Supermarket"] != "Warehouse"]
    # Set index for dictionary conversion
    df = df.set_index('Supermarket')
    # Add columns for use later
    df['RoutesVisitedBy'] = 0
    df['RoutesNeeded'] = 1

    # Create 2 dictionaries, one for the list of routes that visit each store, the other to limit the number of trucks visiting a store
    stores = df.RoutesVisitedBy.to_dict()
    demands = df.RoutesNeeded.to_dict()

    # Turn the values of stores into empty lists
    for i in stores.keys():
        stores[i] = []

    # Fill in for each route and store, whether that store is visited by the route
    for route in routes.Stops:
        for store in stores.keys():
            if store in route:
                stores[store].append(1)
            else:
                stores[store].append(0)

    temp = stores.copy()
    # Removes any supermarkets if they are not visited by any route
    # Impossible to fulfill constraints would be created if a store is not visited by any route
    for store in stores.keys():

        missing_stores = [
            store
            for store, route_visits in stores.items()
            if not any(route_visits)
        ]

        if missing_stores:
            raise ValueError(f"No feasible route generated for: {', '.join(missing_stores)}")

    stores = temp
    return stores, demands

def createLP(routeNames, ownedCosts, leasedCosts, patterns, demands):
    '''
    Create and solve the linear program

    :param routeNames: Names of routes
    :param ownedCosts: cost of operating route with an owned truck
    :param leasedCosts: cost of operating route with a leased truck
    :param patterns: dictionary containing the patterns
    :param demands: dictionary containing the demands

    '''
    # Create a list of the stores involved
    storesList = list(demands.keys())
    # Create a list of all the stores routing involvements
    routes = list(patterns.values())

    # Convert to a dictionary
    routes = makeDict([storesList, routeNames], routes, 0)

    # Create decision variables for each route using an owned truck and a hired truck
    routeNormal = LpVariable.dicts('Route_Owned', routeNames, 0, 1, LpBinary)
    routeLinfox = LpVariable.dicts('Route_Leased', routeNames, 0, 1, LpBinary)

    # Create decision variable for if hired trucks are required
    truckHire = LpVariable('Truck_Hire', 0, 1, LpBinary)

    # Set up LP
    prob = LpProblem('Routing_Problem_Test', LpMinimize)

    # Objective function
    prob += lpSum(routeNormal[i] * ownedCosts[i] + routeLinfox[i] * leasedCosts[i] for i in routeNames), 'Routing Cost'

    # Constraints so each store is visited once
    for i in storesList:
        prob += lpSum((routeNormal[j] + routeLinfox[j]) * routes[i][j] for j in routeNames) == demands[i], f'Ensure {i}'

    # Constraint so we don't use more trucks than we own
    prob += lpSum(routeNormal[i] for i in routeNames) <= numberTrucksAvailable, 'Number of Trucks'

    # Constraint to prevent hiring trucks before we run out
    # prob += lpSum(routeLinfox[i] for i in routeNames) - 99999 * truckHire <= 0, 'Hired Trucks Bound'

    # Constraint to check if we have used all our trucks
    # prob += numberTrucksAvailable * truckHire - lpSum(routeNormal[i] for i in routeNames) <= 0, 'Hired Trucks Toggle'

    # Constraint to prevent a route using both an owned truck and hired truck
    for i in routeNames:
        prob += routeNormal[i] + routeLinfox[i] <= 1, f'One truck per route {i}'

    prob.writeLP('Route_base_sat.lp')

    prob.solve()

    print('Status:', LpStatus[prob.status])

    print('Routing Costs = ', value(prob.objective))

    # Save solution as CSV
    saveToCSV(prob,"LpSolution_base_sat.csv")

def createLPshedding(routeNames, ownedCosts, leasedCosts, patterns, demands):
    '''
    Create and solve the linear program considering the 20% of stores that can have demand shed

    :param routeNames: Names of routes
    :param ownedCosts: cost of operating route with an owned truck
    :param leasedCosts: cost of operating route with a leased truck
    :param patterns: dictionary containing the patterns
    :param demands: dictionary containing the demands

    '''
    # Create a list of the stores involved
    storesList = list(demands.keys())

    # Set penalty for shedding demand based on type of store
    shedPenalty = {}
    for store in storesList:
        if "Pak 'n Save" in store:
            shedPenalty[store] = 1500
        else:
            shedPenalty[store] = 800
    # Create a list of all the stores routing involvements
    routes = list(patterns.values())

    # Convert to a dictionary
    routes = makeDict([storesList, routeNames], routes, 0)

    # Create decision variables for each route using an owned truck and a hired truck
    routeNormal = LpVariable.dicts('Route_Owned', routeNames, 0, 1, LpBinary)
    routeLinfox = LpVariable.dicts('Route_Leased', routeNames, 0, 1, LpBinary)

    #Create binary decision variables for each store to determine whether demand is 'shed'
    shedStore = LpVariable.dicts('Store_Shed', storesList,0,1,LpBinary)

    # Create decision variable for if hired trucks are required
    truckHire = LpVariable('Truck_Hire', 0, 1, LpBinary)

    # Set up LP
    prob = LpProblem('Routing_Problem_Test', LpMinimize)

    # Objective function
    prob += (lpSum(routeNormal[i] * ownedCosts[i] + routeLinfox[i] * leasedCosts[i] for i in routeNames)
             + lpSum(shedStore[i] * shedPenalty[i] for i in storesList)), 'Routing Cost'

    # Constraints so each store is visited once unless demand shed
    for i in storesList:
        prob += (lpSum((routeNormal[j] + routeLinfox[j]) * routes[i][j] for j in routeNames) + shedStore[i]
                 == demands[i], f'Ensure or shed {i}')

    # Constraint so we don't use more trucks than we own
    prob += lpSum(routeNormal[i] for i in routeNames) <= numberTrucksAvailable, 'Number of Trucks'

    # Constraint to prevent hiring trucks before we run out
    # prob += lpSum(routeLinfox[i] for i in routeNames) - 99999 * truckHire <= 0, 'Hired Trucks Bound'

    # Constraint to check if we have used all our trucks
    # prob += numberTrucksAvailable * truckHire - lpSum(routeNormal[i] for i in routeNames) <= 0, 'Hired Trucks Toggle'

    # Constraint to prevent a route using both an owned truck and hired truck
    for i in routeNames:
        prob += routeNormal[i] + routeLinfox[i] <= 1, f'One truck per route {i}'

    # Constraint to limit only 20% of stores have demand shed
    maxStoresShed = int(np.floor(0.2 * len(storesList)))
    prob += lpSum(shedStore[i] for i in storesList) <= maxStoresShed, 'Max stores shed'

    prob.writeLP('Route_Shedding_sat.lp')

    prob.solve()

    print('Status:', LpStatus[prob.status])

    print('Total Costs = ', value(prob.objective))

    print("\nStores shed:")

    for i in storesList:
        if shedStore[i].varValue is not None and shedStore[i].varValue > 0.5:
            print(i)
    # Save what stores had demand shed to csv
    shed_data = []
    for i in storesList:
        if shedStore[i].varValue is not None and shedStore[i].varValue > 0.5:
            shed_data.append({
                "Store": i
            })
    shed_output = pd.DataFrame(shed_data)
    shed_output.to_csv(
        "ShedStores_sat.csv",
        index=False
    )

    # Save overall solution as CSV
    saveToCSV(prob,"LpSolution_shedding_sat.csv")

def saveToCSV(solution, fileName = 'LpSolution.csv'):

    '''
    Saves the output of the linear program to a CSV file

    :param solution: The solved linear program

    :param fileName: name of the file to save to

    '''



    usedRoutes = []

    # Loop through all decision variables
    for v in solution.variables():
        if v.varValue is not None and v.varValue > 0.5:
            # If the route is used, split the name to retrieve the route number
            name = v.name.split('_')

            # Append to list of used routes
            if len(name) == 3:
                usedRoutes.append(name)

    data = []

    # Create DataFrame for each used route containing the name of the route and the truck used on the route
    for i in usedRoutes:
        data.append({
            'Route': routes.loc[int(i[2])].Route,
            'Truck': i[1]
        })

    output = pd.DataFrame(data)

    output.to_csv(fileName, index=False)


routes = prepCosts('Data/saturday_routes.csv')
stores, demands = prepPatterns('Data/FoodstuffsLocations.csv', routes)

# Get the names and costs of each route
names = list(routes.Route.to_dict().keys())
ownedCosts = routes.Cost.to_dict()
leasedCosts = routes.LinfoxCost.to_dict()

createLP(names, ownedCosts, leasedCosts, stores, demands)
createLPshedding(names, ownedCosts, leasedCosts, stores, demands)

