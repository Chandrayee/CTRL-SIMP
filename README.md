# CTRL-SIMP

This repo contains the Med-EASi dataset and models for controllable medical text simplification. The models are released with the first ever medical dataset with fine-grained annotations of elaboration, replacement, deletion and insertion. The model is trained in a multi-angle fashion like MACAW (https://github.com/allenai/macaw), so that, in addition to simplifying the medical text and can also separately predict the content of the expert text that are likely to be difficult for people to understand. 

## Models
We add two types of controllability into text simplification, by using a multi-angle training approach. 
- **CTRL-SIM_ip: position-aware**
- **CTRL-SIM: position-agnostic**
The models are trained with python version 3.8.10 and torch 1.10 (cu-11.3). The model using T5-large was trained on two CUDA devices (GeForce RX 3080 each).

To train the model:
* make appropriate changes to the data path, model path and CUDA devices
* make sure NLTK and pandas installations are working
* run training.py 

Model arguments:
* --ip_ann: bool, default = True, include in place annotated data Sa and Ea
* --one_slot: bool, default = False, train only on in place annotated data (Ea->Sa)
* --shuffle: bool, default = True, shuffle training data to avoid the same examples in one batch

The evaluation functions are in model.py.

## Dataset
To use the dataset from Huggingface use do the following:

```
from datasets import load_dataset
dataset = load_dataset("cbasu/Med-EASi")
```

OR

git lfs install
git clone https://huggingface.co/datasets/cbasu/Med-EASi
