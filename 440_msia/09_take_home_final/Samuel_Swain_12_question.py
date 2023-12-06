import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load your data from Excel
# The header is in the first row, so we don't need 'header=None'
data = pd.read_excel('Final Exam MSiA Data for Students.xlsx') # You need to edit the excel file so the table is top left aligned

# Initialize the model
model = gp.Model("Investment Optimization")

# Create decision variables for each project
projects = range(data.shape[0])  # Number of projects
x = model.addVars(projects, vtype=GRB.BINARY, name="Invest")

# Set objective: Maximize the total NPV
# Using column names after setting the header row
model.setObjective(gp.quicksum(x[i] * data.at[i, 'Total NPV ($ million)'] for i in projects), GRB.MAXIMIZE)

# Add budget constraint: Sum of cash outlay should not exceed $200 million
# Using column names after setting the header row
model.addConstr(gp.quicksum(x[i] * data.at[i, 'Cash Outlay ($ million)'] for i in projects) <= 200, name="Budget")

# Balancing constraint: Assume you want to keep the investment between the divisions as even as possible
# Using column names after setting the header row
total_investment_div1 = gp.quicksum(x[i] * data.at[i, 'Division 1 NPV ($ million)'] for i in projects)
total_investment_div2 = gp.quicksum(x[i] * data.at[i, 'Division 2 NPV ($ million)'] for i in projects)
model.addConstr(total_investment_div1 - total_investment_div2 <= 10, name="Balance_Div1")
model.addConstr(total_investment_div2 - total_investment_div1 <= 10, name="Balance_Div2")

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    print("Optimal investment strategy:")
    for i in projects:
        if x[i].X > 0.5:  # If the project is selected
            print(f"Invest in project {data.at[i, 'Investment Option']}")
