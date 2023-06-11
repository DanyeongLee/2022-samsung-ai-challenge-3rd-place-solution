______________________________________________________________________

<div align="center">

# 2022 Samsung AI Challenge (Materials Discovery)
3rd Place Solution (Team cgu)  
[presentation](/imgs/dacon_samsung2022_cgu.pdf)

</div>


## Team Members
구정현, 이단영, 김상엽

## Solution Overview
![overview](/imgs/structure.png)

- We adopted GeoGNN architecture from [Fang et al., 2022](https://www.nature.com/articles/s42256-021-00438-4) as a base molecule encoder.
- We revised GeoGNN to appropriately model the "difference" between two energy states of molecules.


## Running
### Downloading dataset
- Training and test data can be downloaded from [2022 Samsung AI Challenge (Materials Discovery)](https://dacon.io/competitions/official/235953/data).  
- We assume that you appropriately downloaded the dataset into 'data' directory.

### Training single model
```
python train.py configs/gem1.yaml 
```
- You can train your model using default hyperparameters we used in this competition with above command.  
- Trained model checkpoints and submission files (test_preds.csv) will be saved in 'outputs' directory. You can directly submit the csv file.
- Just a single model achieved high performance (private LB score: 0.65 ~ 0.7), but for further improvement, we used stacking ensemble.

### Stacking ensemble
```
bash run.sh
```
- Running above command will run 10-fold CV of total 12 models (4 different hyperparameters, 3 different seeds). This may take a long time.
- If it ran well, prediction files (csv) on validation set of each fold and test set would have been created.
- By running codes in 'stack_ensemble.ipynb', you can train xgboost models on stacked dataset and generate the ensembled submission file ('outputs/ensembled_submission.csv').
