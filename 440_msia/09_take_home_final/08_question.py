from gurobipy import Model, GRB

# Create a new model
m = Model("ComplexCo")

# Create variables for the amount to purchase from each supplier
A = m.addVar(vtype=GRB.INTEGER, name="A")
B = m.addVar(vtype=GRB.INTEGER, name="B")
C = m.addVar(vtype=GRB.INTEGER, name="C")

# Set objective: Minimize cost
m.setObjective(1 * A + 1.2 * B + 1.5 * C, GRB.MINIMIZE)

# Add demand constraint: Total units purchased = 170
m.addConstr(A + B + C == 170, "demand")

# Add supplier limits
m.addConstr(A <= 100, "limit_A")
m.addConstr(B <= 100, "limit_B")
m.addConstr(C <= 100, "limit_C")

# Add contract constraints
m.addConstr(B >= 2 * A, "contract_B_A")  # At least twice as much from B as from A
m.addConstr(C >= A + B, "contract_C")    # At least as much from C as from A and B combined

# Optimize the model
m.optimize()

# Print solution
if m.status == GRB.OPTIMAL:
    print('Optimal solution found:')
    print(f'A: {A.X} units, B: {B.X} units, C: {C.X} units')
    print(f'Total cost: ${m.objVal} million')
else:
    print('No optimal solution found')
