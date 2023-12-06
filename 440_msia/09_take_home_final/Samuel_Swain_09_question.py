from gurobipy import Model, GRB

# ----- Settings Section -----
# Upper bounds for wheat and corn availability
upper_bound_wheat = 500
upper_bound_corn = 1000

# Ingredient costs
cost_wheat = 0.5
cost_corn = 0.45
cost_alfalfa = 0.4

# Selling prices
price_feed1 = 1.5
price_feed2 = 1.2

# Composition requirements
min_wheat_corn_percentage_feed1 = 0.75
min_wheat_corn_percentage_feed2 = 0.25
min_wheat_to_corn_ratio_feed1 = 2
min_wheat_to_corn_ratio_feed2 = 1.25

# Minimum quantity of wheat and corn in each feed
min_wheat_each_feed = 100
min_corn_each_feed = 100
# ----------------------------

# Create a new model
m = Model("Feedco")

# Add variables with upper bounds for wheat and corn
w1 = m.addVar(name="w1", ub=upper_bound_wheat)
c1 = m.addVar(name="c1", ub=upper_bound_corn)
a1 = m.addVar(name="a1")
w2 = m.addVar(name="w2", ub=upper_bound_wheat)
c2 = m.addVar(name="c2", ub=upper_bound_corn)
a2 = m.addVar(name="a2")
total_weight_feed1 = m.addVar(name="total_weight_feed1", lb=0, ub=upper_bound_wheat + upper_bound_corn)
total_weight_feed2 = m.addVar(name="total_weight_feed2", lb=0, ub=upper_bound_wheat + upper_bound_corn)

# Set the objective: Maximize Profit
profit = price_feed1 * (w1 + c1 + a1) - cost_wheat * w1 - cost_corn * c1 - cost_alfalfa * a1 \
       + price_feed2 * (w2 + c2 + a2) - cost_wheat * w2 - cost_corn * c2 - cost_alfalfa * a2
m.setObjective(profit, GRB.MAXIMIZE)

# Add constraints
# Ingredient limits
m.addConstr(w1 + w2 <= upper_bound_wheat, "wheat_limit")
m.addConstr(c1 + c2 <= upper_bound_corn, "corn_limit")

# Feed composition constraints
m.addConstr(w1 + c1 >= min_wheat_corn_percentage_feed1 * (w1 + c1 + a1), "min_wheat_corn_in_feed1")
m.addConstr(w2 + c2 >= min_wheat_corn_percentage_feed2 * (w2 + c2 + a2), "min_wheat_corn_percentage_feed2")

# Wheat to corn ratio in Feed 1
m.addConstr(w1 >= min_wheat_to_corn_ratio_feed1 * c1, "wheat_to_corn_ratio_feed1")
m.addConstr(w2 >= min_wheat_to_corn_ratio_feed2 * c2, "min_wheat_to_corn_ratio_feed2")

# Minimum wheat and corn in both feeds
m.addConstr(w1 >= min_wheat_each_feed, "min_wheat_in_feed1")
m.addConstr(c1 >= min_corn_each_feed, "min_corn_in_feed1")
m.addConstr(w2 >= min_wheat_each_feed, "min_wheat_in_feed2")
m.addConstr(c2 >= min_corn_each_feed, "min_corn_in_feed2")

# Total composition of each feed type
m.addConstr(w1 + c1 + a1 <= total_weight_feed1, "total_feed1")
m.addConstr(w2 + c2 + a2 <= total_weight_feed2, "total_feed2")

# Non-negativity constraints
m.addConstr(w1 >= 0, "w1_non_negative")
m.addConstr(c1 >= 0, "c1_non_negative")
m.addConstr(a1 >= 0, "a1_non_negative")
m.addConstr(w2 >= 0, "w2_non_negative")
m.addConstr(c2 >= 0, "c2_non_negative")
m.addConstr(a2 >= 0, "a2_non_negative")

# Optimize the model
m.optimize()

# Print solution
if m.status == GRB.OPTIMAL:
    print('Optimal solution found:')
    print(f'Feed 1 - Wheat: {w1.X}, Corn: {c1.X}, Alfalfa: {a1.X}')
    print(f'Feed 2 - Wheat: {w2.X}, Corn: {c2.X}, Alfalfa: {a2.X}')
    print(f'Total Profit: ${m.objVal}')
else:
    print('No optimal solution found')
