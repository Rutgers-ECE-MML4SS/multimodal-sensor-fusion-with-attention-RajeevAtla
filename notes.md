# notes

## environment setup
- conda is really slow, probably significantly contributes to the 30 min time limit -> use uv instead

## data
- data was downloaded from kaggle - https://www.kaggle.com/datasets/diegosilvadefrana/fisical-activity-dataset/
- the original csv file is too large to push to GitHub, so we'll do some preprocessing and splitting before pushing it
- all transient values were removed (see data/readme.pdf - taken from source) - this isn't significant to our problem, and we can focus on classifying current stable activities
- `data/preprocess.py` was then used to split up the data based on subjectID (see data/people_splits)
- cleaning and splitting ensures that all data is < 100 MB GitHub limit

## formatting/linting/type checking
- formatting and linting done with ruff
- type checking done with ty - need to setup
