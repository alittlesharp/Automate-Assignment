Task List is pulled directly from previous assignment sheets with a few notable changes. 
- Removed assignments from Gina's team. That includes Recovery Med files. 
- Removed odd assignments and 2nd looks***

***Reduce total minutes of these associates by the relevant estimated minutes. 
Added capacity to associates. Specialized capacity for those with 2nd looks, NRO, and other designated responsibilities.
A better(?) option that to do the following: add a preload:
#preload = {"Bryant": 240.0}
and then later add the preload code to the R distribution
#for r in R:
#   workload_r = pl.lpSum(m[t] * x[t][r] for t in T)  # NEW minutes
#   effective = workload_r + preload.get(r, 0.0)
#   model += W_max >= effective
#   etc.


***
Currently, the plan is to make a linear programming model for balanced reviewer assignment.
I'm aiming to assign tasks to reviewers to maximize fit (expertise, rapport) while balancing workload. 
*LP formulation and a working Python implementation using PuLP.

I used a code from copilot, but to make sure it worked as intended, I went down every single line, inquired what each
part did. Then I ran tests for every step.