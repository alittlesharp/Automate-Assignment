import pandas as pd
import pulp as pl

#load tasks.csv
tasks = pd.read_csv("tasks.csv")
# print(tasks.head())

# Define reviewers list (from your csv file), the set piece ensures a unique element
reviewers = sorted(set(tasks.get("reviewer", [])))
#print(reviewers)

#list mapping dictionary of csv file where keys map to a list of associated values.
reviewer_expertise = tasks.groupby('reviewer')['work_type'].unique().apply(sorted).to_dict()
#print(reviewer_expertise)

#finds all the unique combinations of rapport from dataframe (via tasks csv)
rapport_combos = tasks[['reviewer', 'manager']]
spec_rapport_combos = rapport_combos.drop_duplicates()

#convert the unique combinations to a dictionary, where keys are tuples of reviewer and manager and values are True
#dictionary past_experience ONLY contain entries for combinations present in csv file
past_experience = {}
for row in spec_rapport_combos.itertuples(name='Rapports'):
    past_experience[(row.reviewer, row.manager)] = True
#print(past_experience)

# Weights
B_RAPPORT = 5
P_SPREAD = 2
LAMBDA_BAL = 0.1

# Build index sets, T will hold row identifier but no data, R holds the unique list of reviewers
T = list(tasks.index)
R = reviewers

# Parameters, making dictionaries of specific values from task csv, T from the iterable index set above
m = {t: float(tasks.loc[t, "est_mins"]) for t in T}
mgr = {t: tasks.loc[t, "manager"] for t in T}
wtype = {t: tasks.loc[t, "work_type"] for t in T}
#print(m)
#print(mgr)
#print(wtype)

# *Added*--- Feasibility check: tasks with zero eligible reviewers ---
for t in T:
    eligible = [r for r in R if wtype[t] in reviewer_expertise.get(r, [])]
    if not eligible:
        print(f"WARNING: Task {t} ({wtype[t]}) has NO eligible reviewers!")

# Suitability, that is it should return weight or preference if there is experience. The false in the get method
# is saying if the key is found it returns the key's value, but if it is NOT found it returns the default value (False)
def suitability(t, r):
    if past_experience.get((r, mgr[t]), False):
        return B_RAPPORT
    else:
        return 0

s = {(t, r): suitability(t, r) for t in T for r in R}
#print(s)

# Model (Note to self: Linear Programming is incredible and it terrifies me.)
model = pl.LpProblem("BalancedReviewerAssignment", pl.LpMaximize)

# Decision vars, binary is yes or no, removed "lowBound=0, upBound=1" since it felt redundant
x = pl.LpVariable.dicts("x", (T, R), cat="Binary")

# HARD CONSTRAINT: Expertise gating
for t in T:
    wt = wtype[t]
    for r in R:
        if wt not in reviewer_expertise.get(r, []):
            x[t][r].upBound = 0  # disallow assignment, funny way of saying it

# Workload bounds, holds the optimization variable, cat = continuous because it is not defined
W_max = pl.LpVariable("W_max", lowBound=0)
W_min = pl.LpVariable("W_min", lowBound=0)

# Creating capacity dictionary, were a reviewer key has none attributed to it.
# Hard constraint can be made by updating capacity dict with a specific limit
capacity = {r: None for r in R}

# Set remaining capacity for Action reviews out of 5900. -550 for Chris, Tim, Sadnan for other tasks.
#special_caps = {
#    "Bryant": 5315,     # less 2nd looks and NRO cap (225)
#    "Girton": 5570,     # less 2nd looks
#    "Harding": 5350,
#    "Marth": 5780,      # less 2nd looks
#    "Mehanovic": 5350,
#    "Mock": 5780,       # less 2nd looks
#    "Perry": 3172,      # less NRO cap (2728)
#    "Rulapaugh": 4599,  # less 2nd looks and NRO cap (1091)
#    "Skelly": 4762,     # 5350 less NRO cap (588)
#    "Verhest": 5780,    # less 2nd looks
#}

# Update the capacity dict for only those listed
#for r, cap in special_caps.items():
#    if r in capacity:
#        capacity[r] = float(cap)

# Preload code. *IMPORTANT* USED ONLY IF CAPACITY ISN'T USED.
preload = {
    "Bryant": 585,
    "Girton": 330,
    "Harding": 550,
    "Marth": 120,
    "Mehanovic": 550,
    "Mock": 120,
    "Perry": 2728,
    "Rulapaugh": 1301,
    "Skelly": 1138,
    "Verherst": 120
}

# Manager unique list. Manager spread vars, removed "lowBound=0, upBound=1" since it felt redundant
# y is a dictionary with (managers, R) as key and 0 or 1 as value, it is specific to PuLP
managers = sorted(tasks["manager"].unique())
y = pl.LpVariable.dicts("y", (managers, R), cat="Binary")

# Objective, maximize expression with three competing priorities. (1) Favoring rapport. (2) Minimize workload imbalance.
# (3) Reduce manager/reviewer spread.
model += (
    pl.lpSum(s[(t, r)] * x[t][r] for t in T for r in R)
    - LAMBDA_BAL * (W_max - W_min)
    - P_SPREAD * pl.lpSum(y[mgr_][r] for mgr_ in managers for r in R)
)

# Constraints
# Each task is assigned only once.
for t in T:
    model += pl.lpSum(x[t][r] for r in R) == 1

#summing time of each task (m[t]), min max tracking will be the total workloads among all
#reviewers in final solution, if capacity specified (not none), a hard constraint is added
for r in R:
    workload_r = pl.lpSum(m[t] * x[t][r] for t in T)
    effective = workload_r + preload.get(r, 0.0)
    model += W_max >= effective
    model += W_min <= effective
#    if capacity[r] is not None:
#        model += workload_r <= capacity[r]

# This does not limit the spread. Instead, it is validating the data.
for mgr_ in managers:
    tasks_of_mgr = [t for t in T if mgr[t] == mgr_]
    for r in R:
        for t in tasks_of_mgr:
            model += y[mgr_][r] >= x[t][r]

# Solve. Hold on to your butts. With time limit because otherwise it will go on forever.
# ***THIS IS WHERE I STOPPED THE DEEP DIVE*** Appending some assistance.
print("\nSolving model...")
model.solve(pl.PULP_CBC_CMD(msg=True, timeLimit=60))

# Extract results
assignment = []
for t in T:
    assigned_r = max(R, key=lambda r: pl.value(x[t][r]))
    assignment.append({
        "task_idx": t,
        "work_type": wtype[t],
        "manager": mgr[t],
        "qa_reviewer": assigned_r,
        "est_mins": m[t],
        "score": s[(t, assigned_r)]
    })

result_df = pd.DataFrame(assignment)

# Print results
print("\nObjective value:", pl.value(model.objective))
print("\nAssignments:")
print(result_df)

# Workload summary
wl = (
    result_df.groupby("qa_reviewer")["est_mins"]
    .sum()
    .reset_index()
    .rename(columns={"est_mins": "total_mins"})
)
print("\nWorkload Summary:")
print(wl)

# Save assignments to CSV
result_df.to_csv("new_assignments.csv", index=False)

# Save workload summary to CSV
wl.to_csv("new_workload_summary.csv", index=False)

print("\nFiles saved: new_assignments.csv and new_workload_summary.csv")