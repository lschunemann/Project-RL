## Training
```
python3 main.py --path PATH/TO/train.xlsx
```

## Validation
```
python3 main.py --path PATH/TO/validate.xlsx
```

## Running different policies
The default policy is PPO, you can run different policies by passing the --agent argument:

```
python3 main.py --path PATH/TO/validate.xlsx --agent tabular_double_q
```


## Visualization
The visualization is currently only supported for Tabular Q agents
It can be rerun by changing the path to the q table in the visualize_q.ipynb jupyter notebook


