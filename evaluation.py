from collections import defaultdict
import re
import random
from preprocessing import load_data
from model import load_model, run_model, run_model_with_outputs, get_eval_data, run_macaw
import torch
import pandas as pd
import pickle
import json
import numpy as np
from utils import GENERATOR_OPTIONS_DEFAULT
import argparse
import textwrap
from rouge_score import rouge_scorer

def run_evaluation(eval_data, model_dict):
    eval_res = []
    eval_gen = []
    rouge_score = []
    eval_loss = 0.
    for _, (input_string, output_strings) in enumerate(eval_data):
        res, loss = run_model_with_outputs(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], input_string, output_strings)
        res_gen = run_model(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], input_string, GENERATOR_OPTIONS_DEFAULT)
        example_rouge = []
        for i in range(len(res_gen['raw_output_list'])):
            output = res[i]['output_raw']
            gen = res_gen['raw_output_list'][i]
            example_rouge.append(rouge(output, gen))
        rouge_score.append(example_rouge)
        eval_res.append(res)
        eval_gen.append(res_gen)
        eval_loss += loss
    eval_loss /= len(eval_data)
    print('average loss on eval data: {}'.format(eval_loss))
    eval = {'eval_res': eval_res, 'eval_gen': eval_gen, 'eval_loss': eval_loss}
    return eval

def testing_run_macaw(eval_data):
    for input, outputs in eval_data:
        print("\n\n-------------new example------------------")
        print("raw inputs: ", input, '\n')
        res = run_macaw(input, outputs, model_dict=model_dict)
        #print(res)

def rouge(output, gen):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(output, gen)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type = int, default = 1)
    args = parser.parse_args()
    model_path = './models/model_' + str(args.chkpt) + '.hf'
    tokenizer_path = 't5-small'
    model_dict = load_model(model_name_or_path=model_path, tokenizer_path = tokenizer_path, cuda_devices = [0])
    
    crowdsourced_data = pd.read_csv("./Datasets/annotated_data/annotated_data.csv")
    crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
    textpairs = [[x,y] for x,y in zip(crowdsourced_data['Expert'], crowdsourced_data['Annotation'])]
    
    eval_data = textpairs[-3:]
    print("There are {} eval text pairs".format(len(eval_data)))
    eval_pairs, all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = load_data(eval_data, eval=True)
    
    eval_data = get_eval_data(eval_pairs, slots_eval, all_annotations_eval)
    testing_run_macaw(eval_data)
    #eval = run_evaluation(eval_pairs, slots_eval, all_annotations_eval, model_dict)