import json
import argparse
import os
import metrics
import time

def parse_data(dataset, datatype):
    file = "./Datasets/" + dataset + "/" + datatype + ".txt"
    print(file)
    if not os.path.exists(file):
        file = "./Datasets/" + dataset + "/" + datatype + ".json"
        if not os.path.exists(file):
            print("wrong directory")

    expert_text = []
    layman_text = []
    expert_concepts = []
    layman_concepts = []

    if dataset == 'msd':
        print("Parsing MSD data")
        if datatype == 'train':
            with open(file, encoding='utf8') as f:
                data = [json.loads(line) for line in f]

        if datatype == 'test':
            with open(file, encoding='utf8') as f:
                data = [json.loads(line) for line in f]

        
        for example in data:
            if example['label'] == 0:
                expert_text.append(example['text'])
                expert_concepts.append([concept['term'] for concept in example['concepts']])
            else:
                layman_text.append(example['text'])
                layman_concepts.append([concept['term'] for concept in example['concepts']])

        
    elif dataset == 'simpwiki':
        print("Parsing Simpwiki data")
        with open(file, encoding='utf8') as f:
            for line in f:
                expert_text.append(line.split('\t')[0])
                layman_text.append(line.split('\t')[1])
        

    elif dataset == 'parasimp':
        print("Parsing Paragraph Simplication data")
        expert_text = []
        layman_text = []
        with open(file) as f:
            dataset = json.load(f)
        for data in dataset:
            expert_text.append(data['abstract'])
            layman_text.append(data['pls'])

                
    else:
        print("Dataset not found")

    return expert_text, layman_text

    

    


    
    




