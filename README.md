## Installation
```
conda create -n project_rl python==3.10
conda activate project_rl
pip install -r requirements.txt
```

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
The full list of available policies can be found in the main.py script


## Visualization
The visualization is currently only supported for Tabular Q agents
It can be rerun by changing the path to the q table in the visualize_q.ipynb jupyter notebook


