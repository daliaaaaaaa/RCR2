from pysat.solvers import Glucose4

# a=1, b=2, c=3, d=4, e=5, f=6

# 1. Define the stratified knowledge base from the image (Page 2)
# Let's map variables to integers: a:1, b:2, c:3, d:4, e:5, f:6
# Negation is represented by negative integers.
strata = [
    {"weight": 0.9,  "clauses": [[1, -2, 4, 5], [-2, 3]]},
    {"weight": 0.78, "clauses": [[2, -3]]},
    {"weight": 0.65, "clauses": [[1, -2, 4, 5], [-2, 3], [-1, 2, 4]]},
    {"weight": 0.60, "clauses": [[1, -2, 4, 5], [2, 3]]},
    {"weight": 0.58, "clauses": [[-1, -2, 4]]},
    {"weight": 0.43, "clauses": [[1, -2, 4, 5], [-2, 3], [1, 2, 4]]},
    {"weight": 0.36, "clauses": [[1, 5], [-2, 3, 6]]},
    {"weight": 0.26, "clauses": [[-1, 4]]},
    {"weight": 0.14, "clauses": [[4]]}
]

def calculate_interest_variable(phi_literal):
    # Initialize bounds as per algorithm 
    n = len(strata)
    l = 0
    u = n - 1
    result_idx = -1

    while l <= u: # Dichotomy principle [cite: 9]
        r = (l + u) // 2 # [cite: 10]
        
        # Create projection Sigma* (all clauses from level 0 to r) [cite: 11, 19]
        with Glucose4() as solver:
            for i in range(r + 1):
                for clause in strata[i]["clauses"]:
                    solver.add_clause(clause)
            
            # Add negation of variable of interest (refutation principle) [cite: 5, 11]
            solver.add_clause([-phi_literal])
            
            if solver.solve(): # If consistent 
                l = r + 1 # Target weight is likely lower (higher index)
            else: # If inconsistent 
                result_idx = r
                u = r - 1
                
    if result_idx != -1:
        return strata[result_idx]["weight"]
    return 0

# Step 3: Calculate value for variable 'd' (literal 4) [cite: 25]
# interest_val = calculate_interest_variable(4)
# print(f"The value Val(d, Sigma) is: {interest_val}")

# generate cvs results for all variables
import csv
variables = [1, 2, 3, 4, 5, 6]  # a, b, c, d, e, f
with open('PossibilityTheory/interest_values.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Variable', 'Interest Value'])
    for var in variables:
        interest_val = calculate_interest_variable(var)
        writer.writerow([var, interest_val])
    print(f"The value Val({var}, Sigma) is: {interest_val}")

