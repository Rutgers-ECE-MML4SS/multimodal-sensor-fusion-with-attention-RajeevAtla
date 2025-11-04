# notes

## environment setup
- conda is really slow, probably significantly contributes to the 30 min time limit -> use uv instead

## data
- data was downloaded from kaggle - https://www.kaggle.com/datasets/diegosilvadefrana/fisical-activity-dataset/
- the original csv file is too large to push to GitHub, so we'll do some preprocessing and splitting before pushing it
- all transient values were removed (see data/readme.pdf - taken from source) - this isn't significant to our problem, and we can focus on classifying current stable activities
- cleaned data is < 100 MB GitHub limit
- heart_rate data is then imputed
- train-val-test split must be 0.7-0.15-0.15

## formatting/linting/type checking
- formatting and linting done with ruff
- type checking done with ty


## running (github actions)
- gh actions gives a pretty slim image - 16 gb ram, 2 vcpu cores (ubuntu-latest)
- can reduce further - 5 gb ram, 1 vcpu (ubuntu-slim); max execution time is 15 minutes however