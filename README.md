# NBA Analytics

This is a project for analyzing historical NBA data, specifically moneylines with the goal of predicting who is going to win every NBA game.

### Included
- Historical dataset (2018-yesterday)
    - Including referee data and computed features ELO and Momentum
    - raw files (.xslx, .csv)
    - DataFrame (through Dataset class)
- Backtesting suite
- Fully Implemented XGBoostClassifier
    - weights in /models
    - ensemble(100) weights in /ensemble
- Jupyter notebook walking through the whole thing

### Usage
clone the repo
cd into the repo root
``` python ./src/main.py -h ```
will give you a run down of all the options. The main ones are
``` --run_backtest ```
and
``` --get_predictions ```
which do pretty much what they say. The backtest one will want a start and end date, otherwise will default to the games so far this season.

#### disclaimer
I don't guarantee or even suggest that any of this works or is correct - it's just an experiment for fun. 

##### secondary disclaimer
JD, if you're reading this - you're a bitch. 
