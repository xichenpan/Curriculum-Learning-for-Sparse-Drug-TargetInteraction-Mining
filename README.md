# 3E-DrugTargetInteraction


### Environment
```
python3.6+
torch
pysmiles==1.0.1
numpy
```
### Data Processor

put ```train.csv``` into ```./data``` folder, then run

```
cd data
python decompose.py
python make_graph_dict.py
```

First generate a more efficent indexed of original dataset ```drug.pkl,target.pkl,pairs.pkl```    
Second generate side information about drug graphs ```element.json,hcount.json```  

