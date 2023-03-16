# CTRL-SIMP

This repo contains the Med-EASi (Medical dataset for Elaborative and Abstractive
Simplification) dataset and models for controllable medical text simplification. 

## Overview
MedEASi is a uniquely crowdsourced and finely annotated dataset for supervised simplification of short medical
texts. Its expert-layman-AI collaborative annotations facilitate controllability over text simplification by marking four
kinds of textual transformations: elaboration, replacement, deletion, and insertion. Med-EASi contains diverse training pairs, each with a different set of textual transformations. We also introduce two T5-large based models for controllable simplification. The models are trained using heterogeneous task descriptions, called multi-angle training, where each input and output are written as a combination of slots, similar to [MACAW] (https://arxiv.org/abs/2109.02593). 

## Models
We finetune T5-large with a combination of prompting and in filling to add two types of controllability into text simplification. 
- **$CTRL-SIM_{ip}$ :** position-aware, where label=annotated data
- **$CTRL-SIM$ :** position-agnostic, where label=content of the expert text that must be edited, type of edit and the unannotated simple text

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
The dataset is available at https://huggingface.co/datasets/cbasu/Med-EASi

To use the dataset from Huggingface datasets library do the following:

```
from datasets import load_dataset
dataset = load_dataset("cbasu/Med-EASi")
```

OR

```
git lfs install
git clone https://huggingface.co/datasets/cbasu/Med-EASi
```

## Citation
```
@article{basu2023med,
  title={Med-EASi: Finely Annotated Dataset and Models for Controllable Simplification of Medical Texts},
  author={Basu, Chandrayee and Vasu, Rosni and Yasunaga, Michihiro and Yang, Qian},
  journal={arXiv preprint arXiv:2302.09155},
  year={2023}
}
```

