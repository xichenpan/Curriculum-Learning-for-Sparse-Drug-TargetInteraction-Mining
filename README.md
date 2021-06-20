# Curriculum Learning for Sparse Drug-TargetInteraction Mining
by Xinyu Xu, Xichen Pan, Haotian Xue, Peiyu Chen

### Environment
```shell
python3.6+
torch
numpy
tensorboardX
# used in test
pysmiles==1.0.1
h5py
```

### Quick Start

Test the model as follows:

```shell
cd \path\to\test.py
python .\test.py --csv_file path\to\csvfile --gpu_id 0
```

Train the model following steps:
1. change pretrain_dir to your own one then put [pretrained protein embedding model](https://github.com/tbepler/protein-sequence-embedding-iclr2019?utm_source=catalyzex.com) in it.
2. put your csvfile into `data` fold then run
```shell
python .\preprocessing\decompose.py
python .\preprocessing\make_graph_dict.py
python .\preprocessing\saveh5.py
```
3. train the model by runing：
```shell
python train.py --save_dir neg3 --gpu_id 0 --neg_rate 3
```
4. Doing curriculum learning by running：
```shell
python train.py --save_dir neg15 --gpu_id 0 --neg_rate 15 --curriculum_weight ./checkpoints/neg3/model.pt
```

### File structure after test

```shell
│  find_threshold.py
│  logits_ex.py
│  README.md
│  requirements.txt
│  result.csv # result file
│  test.py
│  train.py
│  
├─cache
│      model_weight.bin
│      targetfeature.h5
│      
├─ckp
│      submit.h5
│      submit.pt
│      
├─data
│      Dataset.py
│      datautils.py
│      drug.pkl
│      element.json
│      hcount.json
│      pairs.pkl
│      target.pkl
│      test_neg_pairs.pkl
│      test_pos_pairs.pkl
│      
├─models
│      Aggregation.py
│      attn.py
│      convlist.py
│      dt_net.py
│      GraphModels.py
│      labelsmoothing.py
│      
├─preprocessing
│      decompose.py
│      make_graph_dict.py
│      saveh5.py
│      
├─src
│  │  alignment.pyx
│  │  alphabets.py
│  │  fasta.py
│  │  metrics.pyx
│  │  parse_utils.py
│  │  pdb.py
│  │  pfam.py
│  │  scop.py
│  │  transmembrane.py
│  │  utils.py
│  │  __init__.py
│  │  
│  └─models
│          comparison.py
│          embedding.py
│          multitask.py
│          sequence.py
│          __init__.py
│          
└─utils
        general.py
        parser.py
        protein_embedding.py
```
