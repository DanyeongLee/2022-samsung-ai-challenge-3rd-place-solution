______________________________________________________________________

<div align="center">

# 2022 Samsung AI Challenge (Materials Discovery)
3rd Place Solution (Team cgu)

</div>


## Team Members
구정현, 이단영, 김상엽

## Solution Overview
(wip)


## Running
### Download dataset
- Training and test data can be downloaded from [2022 Samsung AI Challenge (Materials Discovery)](https://dacon.io/competitions/official/235953/data).  
- We assume that you appropriately downloaded the dataset into 'data' directory.

### Train single model
```
python train.py configs/gem1.yaml 
```
- You can train your model using default hyperparameters we used in competition with above command.  
- Trained model checkpoints and submission files (test_preds.csv) will be saved in 'outputs' directory. You can directly submit the csv file.
- Just a single model achieved high performance (public score: 0.65 ~ 0.7), but for further improvement, we used stacking ensemble.

### Stacking ensemble
(wip)

