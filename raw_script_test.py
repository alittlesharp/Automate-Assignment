import pandas as pd
import pulp as pl

#load tasks.csv
tasks = pd.read_csv("tasks.csv")

# Define reviewers list (from csv file), set ensures a unique element
reviewers = sorted(set(tasks.get("reviewer", [])))

# Assign sorted dictionary (reviewer, work_type) from tasks CSV to reviewer_expertise var.
reviewer_expertise = tasks.groupby('reviewer')['work_type'].unique().apply(sorted).to_dict()

# Finds all the unique combinations of rapport from dataframe (via tasks csv)
rapport_data = tasks[['reviewer', 'manager']]
associate_rapport = rapport_data.drop_duplicates()

# Convert the unique combinations to a dictionary, where keys are tuples of reviewer and manager and values are True
# Dictionary past_experience ONLY contain entries for combinations present in csv file. See suitability later.
past_experience = {}
for row in associate_rapport.itertuples(name='Rapports'):
    past_experience[(row.reviewer, row.manager)] = True

# In linear programming B, P, and lambda are typical notations for basic, cost (price), and Lagrange multipliers.
# Think of B_Rapport as a constant value we give to relationships that we want to keep up. P_spread is a variable
# related to manager spread (later used to ensure there's only 2 associates per manager). LAMBDA_BAL is used in the
# LP model to optimize results (basically to reduce workload imbalance between associates). A low value means slack.
B_RAPPORT = 5
P_SPREAD = 2
LAMBDA_BAL = 0.1

# Build task index set
task_id = list(tasks.index)

# Parameters, making dictionaries of specific values from task csv
est_min = {t: float(tasks.loc[t, "est_mins"]) for t in task_id}
mgr = {t: tasks.loc[t, "manager"] for t in task_id}
discipline = {t: tasks.loc[t, "work_type"] for t in task_id}
# Assoc parameter dictionary was added here for distribution. Comments could be attached in the same way.
assoc = {t: tasks.loc[t, "associate"] for t in task_id}

# Feasibility check. Tasks with zero eligible reviewers will print a warning.
for t in task_id:
    eligible = [r for r in reviewers if discipline[t] in reviewer_expertise.get(r, [])]
    if not eligible:
        print(f"WARNING: Task {t} ({discipline[t]}) has NO eligible reviewers!")

# Suitability, should return weight or preference if there is experience. The false in the get method means that if
# the key is found it returns the key's value, but if it is NOT found it returns the default value (False).
def suitability(t, r):
    if past_experience.get((r, mgr[t]), False):
        return B_RAPPORT
    else:
        return 0

suit_dict = {(t, r): suitability(t, r) for t in task_id for r in reviewers}

# Establishes model
model = pl.LpProblem("BalancedReviewerAssignment", pl.LpMaximize)

# Decision var, yes or no, removed "lowBound=0, upBound=1", constraint_x used to align expertise
constraint_x = pl.LpVariable.dicts("constraint_x", (task_id, reviewers), cat="Binary")

# HARD CONSTRAINT. Expertise.
for t in task_id:
    t_disc = discipline[t]
    for r in reviewers:
        if t_disc not in reviewer_expertise.get(r, []):
            constraint_x[t][r].upBound = 0  # prevents assignment

# Workload bounds, holds optimization variable for model calculation
W_max = pl.LpVariable("W_max", lowBound=0)
W_min = pl.LpVariable("W_min", lowBound=0)

# Preload code. Manually adjusted currently to account for special disc (2nd look, recovery, etc.) and projects.
preload = {
    "Bryant": 360,
    "Girton": 120,
    "Harding": 360,
    "Marth": 120,
    # "Mehanovic": 4750,
    "Mock": 120,
    "Perry": 2584,
    "Rulapaugh": 1261,
    "Skelly": 894,
    "Verhelst": 288,
}

# Manager unique list. Manager spread var, removed "lowBound=0, upBound=1" since redundant
# constraint_y is a dictionary with (managers, R) as key and 0 or 1 as value, constraint_y used to align rapport
managers = sorted(tasks["manager"].unique())
constraint_y = pl.LpVariable.dicts("constraint_y", (managers, reviewers), cat="Binary")

# Maximize expression with three competing priorities. (1) Favor rapport. (2) Minimize workload imbalance.
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