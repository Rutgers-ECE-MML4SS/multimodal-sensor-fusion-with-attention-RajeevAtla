# notes

## environment setup
- conda is really slow, probably significantly contributes to the 30 min time limit -> use uv instead

## data

- train-val-test split must be 0.7-0.15-0.15
- `data/preprocess.py` merges every PAMAP2 `.dat`, drops `activity_id=0`, and writes an aligned CSV after verifying each row has the 56-field schema.
- Heart-rate gaps are handled per subject via forward-fill/back-fill followed by a 25-sample rolling median to keep the signal smooth before training consumes it.

## formatting/linting/type checking
- formatting and linting done with ruff
- type checking done with ty


## running (github actions)
- gh actions gives a pretty slim image - 16 gb ram, 2 vcpu cores (ubuntu-latest)
- can reduce further - 5 gb ram, 1 vcpu (ubuntu-slim); max execution time is 15 minutes however
