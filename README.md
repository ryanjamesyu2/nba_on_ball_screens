# NBA On Ball Screens
A repository to store code related to the 36-660 Sports Analytics Project, which evaluates the effect on on-ball screens in the NBA

## Data

The data used is proprietary data from FTN, bought by the Carnegie Mellon Sports Analytics Center. As such, the data used is not included in this repository.

The file `screen_dataset_creation.qmd` contains the code used to read the raw data and compile a dataset to be used in the project. It also contains some EDA.

## Models

The `bayesian_hierarchical.qmd` was written by Ryan Yu. It contains additional EDA, as well as model fitting and comparison for the Bayesian Hierarchical model to predict the Bernoulli response of a shot going in or not.
