# MaxDivide
MaxDivide is a tool for dividing data into several subsets with maximally different multidimensional distributions.

Such division should aid in setting up validation for the selection of well-generalizing and robust models.

The difference between distributions is calculated as a combination of several metrics:
- Average correlation
- Joint entropy
- Average joint moment
- Generalized variance
- Average Mahalanobis distance
- Trace of the covariance matrix
- Maximum eigenvalue of the covariance matrix
- Conditional entropy
- Total Variation Distance
- Energy statistics

MaxDivide is in the early stages of development, so the list may change.

Currently, an implementation based on a genetic algorithm is being developed. In the future, other algorithms such as Hill Climbing, simulated annealing, Bayesian optimization, and possibly other evolutionary algorithms are planned to be added.
