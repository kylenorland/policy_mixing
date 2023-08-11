# Experiments from Chapter 4: Stochastic Policy Mixing (SPM): Applying the SBRL Framework to Hierarchical Reinforcement Learning
Experimental code illustrating the benefits of a novel SBRL based policy mixing approach for reinforcement learning.

## Replication Instructions: Down Right Experiment
1. Clone repository
2. Using Miniconda, install the packages listed at the top of the Policy Mixing/multi_policy.py file.
3. Under "if generate_mode": near line 490, uncomment the type of experiment that you would like to run.
4. Note: "gen_dr_vs_eg" is the one used in the dissertation. New / alterative experiments can be based off of it.
5. Run the multi_policy.py file using python.
6. Results can be found in Policy Mixing/results.
7. Changes to base parameters can be made by altering the "default_run" under default run settings (line 75)

## Replication Instructions: Analytical Experiment
1. Clone repository (If not already cloned)
2. Using Miniconda, install the packages listed at the top of the Analytic Mixing/analytic_multi_policy.py file.
3. Under "if generate_mode": near line 193, uncomment the type of experiment that you would like to run.
4. Note: "gen_analytic_policy_active" is the one used in the dissertation. New / alterative experiments can be based off of it.
5. Run the analytic_multi_policy.py file using python.
6. Results can be found in Analytic Mixing/results.
7. Changes to base parameters can be made by altering the "default_run" under default run settings (line 77)

## Made improvements or have questions?
Contact the author at kylenorland@arizona.edu
