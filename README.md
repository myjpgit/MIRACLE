# MIRACLE: Mining Implicit Relationships with Multi-view Unsupervised Graph Contrastive Learning

This is the source code of paper "MIRACLE: Mining Implicit Relationships with Multi-view Unsupervised Graph Contrastive Learning". 

## REQUIREMENTS
This code requires the following:
* Python==3.7
* PyTorch==1.7.1
* DGL==0.7.1
* Numpy==1.20.2
* Scipy==1.6.3
* Scikit-learn==0.24.2
* Munkres==1.1.4
* ogb==1.3.1

## USAGE
### Step 1: All the scripts are included in the "scripts" folder. Please get into this folder first.
```
cd scripts
```

### Step 2: Run the experiments you want:

\[Weibo Same-city\]Implicit relationships prediction:
```
bash samecity_lp.sh
```
\[MAG Advisor-advisee\]Implicit relationships prediction:
```
bash advisor_lp.sh
```
\[Terrorist Attacks\]Implicit relationships prediction:
```
bash terror_lp.sh
```

