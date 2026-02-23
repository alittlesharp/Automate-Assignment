ASSIGNMENT AUTOMATION
Written by Isaac Perry

This program uses Python's pandas and pulp libraries to run a LP (linear programming) model that distributes
an optimized assignment list. It uses several constraints to determine the suitability of the assignments.
Assignments consider rapport (previous working relationships between reviewer and manager), expertise (previous
work with a specific discipline), manager/reviewer spread (maximum of 2 QA reviewers per manager), and targeting
a balanced spread of workload between QA reviewers.

Prior to running the program, update the preload dictionary in the python script with workload minutes that are
NOT related to Action reviews. You can get the minutes directly from the prior assignment sheet once you factor
out the Action reviews and add in upcoming 2nd look minutes.
The QA reviewers on special projects (currently, Mehanovic, Harding, Skelly) have another 550 minutes preloaded.

***Limitations***
- By design, the program targets Action reviews only. Other work (2nd look reviews, recovery, special projects) are
mostly consistent month to month. While this work could be automated the focus of this program was Action reviews.
- Program depends completely on the task CSV file to work and this file must be manually prepared in the current
state. The shared assignment sheet is different (ie., work_type, associate, manager, reviewer, est_mins). This
means that currently the person running the program must ensure the CSV file is formatted correctly, with the
correct est_mins for the month (varies between open and closed), and then make sure the CSV file reflects updates
that are accurate (possible for QA users to incorrectly fill out prior month's assignment sheet).
- The current design requires the person running this program to update the preload data month to month. This
change is minimal, but the programmer should ensure the minutes they add are accurate before running the model.

***Future Implementations***
- Future implementations may use more CSV files to execute (expertise and rapport files) in addition to task file.
This would allow accessible modification of expertise and rapport without having to modify the task CSV file.
- Preload CSV file or separate preload script could modify preload capacity month to month without having to
modify the main program.