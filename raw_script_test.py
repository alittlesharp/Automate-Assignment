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
rapport_data = tasks[['reviewer', 'manager']]
associate_rapport = rapport_data.drop_duplicates()

#convert the unique combinations to a dictionary, where keys are tuples of reviewer and manager and values are True
#dictionary past_experience ONLY contain entries for combinations present in csv file. See suitability later.
past_experience = {}
for row in associate_rapport.itertuples(name='Rapports'):
    past_experience[(row.reviewer, row.manager)] = True
#print(past_experience)

# In linear programming B, P, and lambda are typical notations for basic, cost (price), and Lagrange multipliers.
# Think of B_Rapport as a constant value we give to relationships that we want to keep up. P_spread is a variable
# related to manager spread (later used to ensure there's only 2 associates per manager). LAMBDA_BAL is used in the
# LP model to optimize results (basically to reduce workload imbalance between associates). A low value means slack.
B_RAPPORT = 5
P_SPREAD = 2
LAMBDA_BAL = 0.1

# Build index sets, T will hold row identifier but no data, R holds the unique list of reviewers
task_id = list(tasks.index)
# R = reviewers

# Parameters, making dictionaries of specific values from task csv, T from the iterable index set above
est_min = {t: float(tasks.loc[t, "est_mins"]) for t in task_id}
mgr = {t: tasks.loc[t, "manager"] for t in task_id}
discipline = {t: tasks.loc[t, "work_type"] for t in task_id}
#Added assoc parameter dictionary for later distribution.
assoc = {t: tasks.loc[t, "associate"] for t in task_id}
#print(m)
#print(mgr)
#print(wtype)

# *Added*--- Feasibility check: tasks with zero eligible reviewers ---
for t in task_id:
    eligible = [r for r in reviewers if discipline[t] in reviewer_expertise.get(r, [])]
    if not eligible:
        print(f"WARNING: Task {t} ({discipline[t]}) has NO eligible reviewers!")

# Suitability, that is it should return weight or preference if there is experience. The false in the get method
# is saying if the key is found it returns the key's value, but if it is NOT found it returns the default value (False)
def suitability(t, r):
    if past_experience.get((r, mgr[t]), False):
        return B_RAPPORT
    else:
        return 0

suit_dict = {(t, r): suitability(t, r) for t in task_id for r in reviewers}
#print(s)

# Establishes model
model = pl.LpProblem("BalancedReviewerAssignment", pl.LpMaximize)

# Decision var, yes or no, removed "lowBound=0, upBound=1", constraint_x used to align expertise
constraint_x = pl.LpVariable.dicts("constraint_x", (task_id, reviewers), cat="Binary")

# HARD CONSTRAINT: Expertise gating
for t in task_id:
    t_disc = discipline[t]
    for r in reviewers:
        if t_disc not in reviewer_expertise.get(r, []):
            constraint_x[t][r].upBound = 0  # prevents assignment

# Workload bounds, holds optimization variable for model calculation
W_max = pl.LpVariable("W_max", lowBound=0)
W_min = pl.LpVariable("W_min", lowBound=0)

# Creating capacity dictionary, were a reviewer key has none attributed to it.
# Hard constraint can be made by updating capacity dict with a specific limit
# capacity = {r: None for r in R}

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

# Preload code. Manually adjusted currently to account for special disc (2nd look, recovery, etc.) and projects.
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

# Manager unique list. Manager spread var, removed "lowBound=0, upBound=1" since redundant
# constraint_y is a dictionary with (managers, R) as key and 0 or 1 as value, constraint_y used to align rapport
managers = sorted(tasks["manager"].unique())
constraint_y = pl.LpVariable.dicts("constraint_y", (managers, reviewers), cat="Binary")

# Objective, maximize expression with three competing priorities. (1) Favor rapport. (2) Minimize workload imbalance.
# (3) Reduce manager/reviewer spread.
model += (
    pl.lpSum(suit_dict[(t, r)] * constraint_x[t][r] for t in task_id for r in reviewers)
    - LAMBDA_BAL * (W_max - W_min)
    - P_SPREAD * pl.lpSum(constraint_y[mgr_][r] for mgr_ in managers for r in reviewers)
)

# Each task is assigned only once.
for t in task_id:
    model += pl.lpSum(constraint_x[t][r] for r in reviewers) == 1

# For each reviewer, assigning task if suitable, summing time for each task, then factoring in preload work and then
# balancing the distribution. This code is running a calculation.
for r in reviewers:
    rev_workload = pl.lpSum(est_min[t] * constraint_x[t][r] for t in task_id)
    effective = rev_workload + preload.get(r, 0.0)
    model += W_max >= effective
    model += W_min <= effective
#    if capacity[r] is not None:
#        model += workload_r <= capacity[r]

# This does not limit the spread. Instead, it is validating data (expertise and rapport).
for mgr_ in managers:
    tasks_of_mgr = [t for t in task_id if mgr[t] == mgr_]
    for r in reviewers:
        for t in tasks_of_mgr:
            model += constraint_y[mgr_][r] >= constraint_x[t][r]

# Solve the model outlined above. Set time limit of 60 sec. Otherwise, it continued to run. 60 secs may be overkill.
print("\nSolving model...")
model.solve(pl.PULP_CBC_CMD(msg=True, timeLimit=60))

# Extract results into list
assignment = []
for t in task_id:
    assigned_r = max(reviewers, key=lambda r: pl.value(constraint_x[t][r]))
    assignment.append({
        "task_idx": t,
        "associate": assoc[t],
        "work_type": discipline[t],
        "manager": mgr[t],
        "qa_reviewer": assigned_r,
        "est_mins": est_min[t],
        "score": suit_dict[(t, assigned_r)]
    })

# Creates table from the list above
results_assign = pd.DataFrame(assignment)

# Print assignment results
print("\nObjective value:", pl.value(model.objective))
print("\nAssignments:")
print(results_assign)

# Workload summary, changes column from estimated minutes to total minutes. Print workload results.
results_workload = (
    results_assign.groupby("qa_reviewer")["est_mins"]
    .sum()
    .reset_index()
    .rename(columns={"est_mins": "total_mins"})
)
print("\nWorkload Summary:")
print(results_workload)

# Save assignments to CSV
results_assign.to_csv("new_assignments.csv", index=False)

# Save workload summary to CSV
results_workload.to_csv("new_workload_summary.csv", index=False)

print("\nFiles saved: new_assignments.csv and new_workload_summary.csv")