#!/usr/bin/env python
# coding: utf-8

# # Load Packages

# In[1]:


from collections import defaultdict
from itertools import permutations, combinations
from time import time

import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd


# # Flow Formulation of VRP

# In[2]:


def vrp_flow_formulation(max_stops=4, max_num_trucks=5):  # How can you estimate max_num_trucks?
    num_nodes = input_data.shape[0]
    num_cust = num_nodes - 1  # the first row in the file is depot
    node_rng = range(num_nodes)
    c_rng = range(1, num_nodes)
    t_rng = range(max_num_trucks)

    model = gp.Model('vrp')
    # Variables
    x = model.addVars(num_nodes, num_nodes, max_num_trucks, vtype=gp.GRB.BINARY, name='x')

    # Constraints
    model.addConstrs((gp.quicksum(x[i, j, k] for i in node_rng for k in t_rng if i != j) == 1
                      for j in c_rng), name='c2')  # exactly one incoming arc
    
    model.addConstrs((gp.quicksum(x[0, j, k] for j in c_rng) == 1
                      for k in t_rng), name='c3')  # each vehicle is used
    
    model.addConstrs((gp.quicksum(x[i, j, k] for i in node_rng if i != j) ==
                      gp.quicksum(x[j, i, k] for i in node_rng if i != j)
                      for j in node_rng for k in t_rng), name='c4')  # flow conservation

    model.addConstrs((gp.quicksum(x[i, j, k] for i in node_rng for j in node_rng if i != j)
                      <= max_stops for k in t_rng), name='stops')  # max num stops

    # MTZ subtour
    u = model.addVars(num_nodes, max_num_trucks, ub=num_cust - 1,
                      vtype=gp.GRB.CONTINUOUS, name='u')  # auxiliary variable 
    model.addConstrs((u[j, k] - u[i, k] + num_cust * x[i, j, k] <= num_cust - 1
                      for i in c_rng for j in c_rng for k in t_rng if i != j), name='sub1')

    # # DFJ subtour
    # all_sets = [tuple(it) for n in range(2, num_nodes - 2) for it in combinations(c_rng, n)]
    #
    # model.addConstrs((gp.quicksum(x[i, j, k] for i in s for j in s if i != j)
    #                   <= len(s) - 1 for s in all_sets for k in t_rng), name='sub')

    """But this way of dealing with subtour is inefficient. 
    The way to do it is to first solve the problem without the subtour constraints. 
    If the solution contains subtour, then the violated subtours are added to the model
    and the problem is solved again. This is handled through callback function.
    If you like to learn more about this or see an example, 
    check gurobi's implementation of TSP which uses callback.
    https://www.gurobi.com/documentation/current/examples/tsp_py.html
    """

    objective = gp.quicksum(
        dist_matrix[i + 1, j + 1] * x[i, j, k]
        for i in node_rng for j in node_rng for k in t_rng if i != j)
    model.setObjective(objective, gp.GRB.MINIMIZE)
    # model.write(model.ModelName + '.lp')

    # Set Model Parameters
    # Check the link below for a list of parameters and their descriptions:
    # https://www.gurobi.com/documentation/current/refman/parameters.html
    model.setParam('OutputFlag', 1)  # Enable or disable solver output
    # model.parameters.mip.tolerances.mipgap = 0.02  # MIP Gap
    # model.Params.timelimit = 10  # Timelimit in seconds
    model.optimize()
    # If you want to learn about gurobi log: https://www.gurobi.com/documentation/current/refman/mip_logging.html
    print(model.status)

    if model.status == gp.GRB.Status.OPTIMAL:
        for v in model.getVars():
            if v.x > 0.5:
                print('%s : %g' % (v.varName, v.x))

        print('Objective Value : %g' % model.objVal)
    else:
        print('Could not find a solution!')


# # Set Partitioning Formulation of VRP
# There are a couple of changes in this version:
# 
# 1. This is a case of Open VRP where we don't require the vehicles to return to the depot.
# 2. Since the depot is the *obvious* first place to start, it's not counted toward the number of stops. In other words, in this implementation, `max_stops` is really maximum number of drops/delivery. So, a 2-stop route, need to leave the depot and stop at two customer location.
# 3. Similarly, since the depot is the *obvious* first place to start, it is not included in reporting the final routes.

# In[3]:


def vrp_sp_formulation(max_stops=3):
    start = time()
    route_dist_dict, route_cost_dict = create_routes(input_data, dist_matrix, max_stops)
    order_route_dict = get_locations_in_each_route(input_data, route_dist_dict)

    model = gp.Model('vrp')
    x = model.addVars(route_dist_dict, vtype=gp.GRB.BINARY, name='x')
    for loc, routes in order_route_dict.items():
        model.addConstr(gp.quicksum(x[route] for route in routes) == 1, f'location_coverage_{str(loc)}')

    # I use total distance. You can use total cost too (it's available for you)
    objective = gp.quicksum(dist * x[route] for route, dist in route_dist_dict.items())
    model.setObjective(objective, gp.GRB.MINIMIZE)
    # model.write(model.ModelName + '.lp')
    model.setParam('OutputFlag', 0)
    model.optimize()
    exe_time = time() - start

    if model.status == gp.GRB.Status.OPTIMAL:
        if PRINT_OPTIMAL_VALUES:
            for v in model.getVars():
                if v.x > 0.5:
                    print('%s : %g' % (v.varName, v.x))

        print('Objective Value : %g' % model.objVal)
    else:
        print('Could not find a solution!')

    print("Execution Time: {}".format(exe_time))
    print("--------------")
    return model.objVal, exe_time


# ## Helper functions
# These are used in the SP formulation of VRP

# In[4]:


# Create the distance matrix between every two locations
def create_distance_matrix(locations):
    distance_matrix = {}
    for ind1, loc1 in locations.iterrows():
        for ind2, loc2 in locations.iterrows():
            dist = calculate_distance_manhattan(loc1['Lat'], loc1['Long'], loc2['Lat'], loc2['Long'])
            distance_matrix[(ind1, ind2)] = dist
    return distance_matrix


# In[5]:


# Calculates distance using Manhattan
def calculate_distance_manhattan(lat1, lon1, lat2, lon2):
    # Calculate the absolute differences
    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)
    
    # Sum up the absolute differences
    return dlat + dlon


# In[6]:


# Calculates distance using Haversine formula
def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    # radius of earth
    from math import radians, cos, sin, asin, sqrt
    r = 3962.173405788

    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = [radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


# In[7]:


# Create all the routes
def create_routes(input_data, dist_matrix, number_of_stops):
    route_distance_dict, route_cost_dict = {}, {}
    for n in range(1, number_of_stops + 1):
        all_permutations = permutations(input_data.iloc[1:].index, n)
        for route in all_permutations:
            tot_dist = dist_matrix[1, route[0]]
            for i in range(n - 1):
                tot_dist += dist_matrix[route[i], route[i + 1]]
            route_distance_dict.update({route: tot_dist})
            route_cost_dict.update({route: calculate_cost(tot_dist)})
    return route_distance_dict, route_cost_dict


# In[8]:


# Calculate the cost of each route
def calculate_cost(route_distance):
    return max(route_distance * 1.5, 450)


# In[9]:


# Create the coefficient matrix that shows the location coverage: get all the routes that visit a location
def get_location_route_matrix(input_data, route_dist_dict):
    location_route_matrix = {}
    for route in route_dist_dict:
        for loc in input_data.iloc[1:].index:
            if loc in route:
                location_route_matrix[loc, route] = 1
            else:
                location_route_matrix[loc, route] = 0
    return location_route_matrix


# In[10]:


# Better version of the above function:
# Create the coefficient matrix that shows the location coverage: get all the routes that visit a location
def get_locations_in_each_route(input_data, route_dist_dict):
    location_route_matrix = defaultdict(list)
    for route in route_dist_dict:
        for loc in input_data.iloc[1:].index:
            if loc in route:
                location_route_matrix[loc].append(route)
    return location_route_matrix


# # Plot Number of Stops vs Time & Objective Value
# This is for SP formulation of VRP

# In[11]:


def plot_vrp_variations(max_drops=4):
    obj_list, exe_time_list = [], []
    rng = range(1, max_drops + 1)
    for n in rng:
        print("Max number of stops: {}".format(n))
        obj_val, exe_time = vrp_sp_formulation(n)
        obj_list.append(obj_val)
        exe_time_list.append(exe_time)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    plt.xticks(range(max_drops + 1))

    ax.plot(rng, obj_list)
    ax.set_ylim(10)
    ax.set_xlabel('Max Number of Stops')
    ax.set_ylabel('Objective Function')

    ax2.plot(rng, exe_time_list, c='orange')
    ax2.set_ylabel('Time (sec)')
    plt.savefig('vrp_comparison.png')
    plt.show()


# # Run Block
# You should play around with these options. Look at the parameters that are hard-coded in each of these functions. 
# If the value of a parameter used in the 10-customer version is very small, it'll lead to an infeasible solution in the 20-customer version.

# In[12]:


# Load the data
PRINT_OPTIMAL_VALUES = True
input_data = pd.read_csv("manhattan_customers20.csv", index_col=0)
dist_matrix = create_distance_matrix(input_data)
vrp_flow_formulation(5)
vrp_sp_formulation(5)
plot_vrp_variations(5)  # default is 4 stops. Try larger numbers and see the effect


# In[ ]:




