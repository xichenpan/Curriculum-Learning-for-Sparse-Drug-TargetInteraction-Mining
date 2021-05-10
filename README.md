# 3E-DrugTargetInteraction


### Environment
```
python3.6+
torch
pysmiles==1.0.1
numpy
tensorboardX
```

### Allen's Code Structure

```
    -------
       |---data
            |---train.csv
            |---drugs.pkl (after pre-processing)
            |---targets.pkl (after pre-processing)
            |---pairs.pkl (after pre-processing)
            |---element.json (after pre-processing)
            |---hcount.json (after pre-processing)
       |---preprocessing
            |---decompose.py
            |---make_graph_dict.py
       |---utils
            |---parser.py
            |---init.py
            |---protein_embedding.py
            |---general.py 
       |---models
            |---GraphModels.py
            |---dt_net.py
       |---src
            code for pretraininig
       |---train.py
       |---README.md
```


### Data PreProcessing
put ```train.csv``` into ```./data``` folder, then run
```
    python ./preprcessing/decompose.py
    python ./preprcessing/make_graph_dict.py
```

First script decompose DT pairs and make train-test-split.   
Drug info are stored in ```./data/drugs.pkl```   
Targets info are stored in ```./data/targets.pkl```   
DT-pairs info are stored in ```./data/pairs.pkl``` in the format of ```List[...[drug_id,target_id,label,isTraingMask]...]```, where ```isTrainingMask``` = 1 for training, 0 for testing.

You should make sure to see the log below for correct split.
```
# TrainPositive = 14284
# TrainNegative = 22759814
# TestPositive = 1572
# TestNegative = 2528401
```

Second script generate ```element.json,hcount.json```


### TODO
* code for fusion
  * conv1d
  * map
* experiment
