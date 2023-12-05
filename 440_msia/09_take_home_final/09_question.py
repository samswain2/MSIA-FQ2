from gurobipy import Model, GRB

# Create a new model
m = Model("Feedco")

# Add variables
w1 = m.addVar(name="w1")
c1 = m.addVar(name="c1")
a1 = m.addVar(name="a1")
w2 = m.addVar(name="w2")
c2 = m.addVar(name="c2")
a2 = m.addVar(name="a2")

# Set the objective: Maximize Profit
profit = 1.5 * (w1 + c1 + a1) + 1.2 * (w2 + c2 + a2) - 0.5 * (w1 + w2) - 0.45 * (c1 + c2) - 0.4 * (a1 + a2)
m.setObjective(profit, GRB.MAXIMIZE)

# Add constraints
# Ingredient limits (sample data: 500 for wheat, 1000 for corn)
m.addConstr(w1 + w2 <= 500, "wheat_limit")
m.addConstr(c1 + c2 <= 1000, "corn_limit")

# Feed composition
epsilon = 0.001  # Small value to enforce strict inequality
m.addConstr(w1 + c1 >= w2 + c2 + epsilon, "feed1_better")
m.addConstr(w1 >= c1 + epsilon, "wheat_valued_in_feed1")
m.addConstr(w2 >= c2 + epsilon, "wheat_valued_in_feed2")

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
