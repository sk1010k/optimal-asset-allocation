## Implementation of "Optimal Asset Allocation using Adaptive Dynamic Programming" (Ralph Neuneier, 1996).

NOTE: RL implementation in this repo is largely base on https://github.com/dennybritz/reinforcement-learning.

### Requirements

Install:
 - Python 3.6
 - Pipenv
 
 Then:
 ```
 pipenv sync
 pipenv shell
 ```

### How to train & test

- For discrete price space:
```
python discrete.py
```

- For continuous price space:
```
python continuous.py
```

- See `demo.ipynb` for visual demo.
