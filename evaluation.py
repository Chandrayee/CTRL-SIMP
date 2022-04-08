from collections import defaultdict
import re
import random
from preprocessing import load_data
from automatic_annotation import get_replacement
from model import model_input_format, load_model, run_model, run_model_with_outputs, get_eval_data, run_macaw, run_generate
import torch
import pandas as pd
import pickle
import json
import numpy as np
from utils import GENERATOR_OPTIONS_DEFAULT
import argparse
import textwrap
from rouge_score import rouge_scorer
import difflib

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
        #print("raw inputs: ", input, '\n')
        res = run_macaw(input, outputs, model_dict=model_dict)
        print([x['output_raw'] for x in res['explicit_outputs']])
        print(res['output_slots_list'])
        
def create_batch_for_generation(eval_data):
    batch_instances = []
    for input, outputs in eval_data:
        instance = {}
        state_dict = {'input':input, 'output_fields':outputs}
        input, outputs, angle = model_input_format(state_dict)
        instance['input'] = input
        instance['angle'] = angle
        for output_string, output_text in outputs:
            xinstance = instance.copy()
            xinstance['output'] = output_string
            xinstance['output_text'] = output_text
            batch_instances.append(xinstance)
    return batch_instances

def batch_for_conditional_gen(eval_data):
    batch_instances = []
    for input, outputs in eval_data:
        instance = {}
        state_dict = {'input':input, 'output_fields':outputs}
        input, outputs, angle = model_input_format(state_dict)
        instance['input'] = input
        instance['angle'] = angle
        i = len(outputs)
        for output_string, output_text in outputs:
            xinstance = instance.copy()
            xinstance['true_output'] = output_string
            xinstance['true_output_text'] = output_text
            batch_instances.append(xinstance)
            i -= 1
            if i > 0:
                updated_input = instance['input'].split(';')
                updated_input = updated_input[1:]
                instance['input'] = ';'.join(updated_input)
                instance['input'] = instance['input'].strip()
                instance['input'] += ' ; ' + output_string
    return batch_instances

def batch_for_conditional_gen_merged_outputs(eval_data):
    batch_instances = []
    for input, outputs in eval_data:
        instance = {}
        state_dict = {'input':input, 'output_fields':outputs}
        input, outputs, angle = model_input_format(state_dict)
        instance['input'] = input
        instance['angle'] = angle
        i = len(outputs)
        output_string = [string for string, _ in outputs]
        if len(outputs) > 1:
            output_string = ' ; '.join(output_string)
        else:
            output_string = output_string[0]
        output_text = [text for _, text in outputs]
        instance['true_output'] = output_string
        instance['true_output_text'] = output_text
        batch_instances.append(instance)
    return batch_instances
    

def run_batch_generation(model, tokenizer, cuda_device, generator_options, batch_instances):
    all_res = run_generate(model, tokenizer, cuda_device, generator_options, batch_instances)
    return all_res

def rouge(output, gen):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(output, gen)
    return scores

def compute_rouge(ar1, ar2):
    rouges = []
    for output, gen in zip(ar1, ar2):
        rouges.append(rouge(output, gen))
        
def compute_diff(ar1, ar2):
    seq_mat = difflib.SequenceMatcher()
    diff = []
    ratio = []
    for output, gen in zip(ar1, ar2):
        seq_mat.set_seq1(output)
        seq_mat.set_seq2(gen)
        ratio.append(seq_mat.ratio())
        diff.append(get_replacement(output, gen))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type = int, default = 1)
    args = parser.parse_args()
    model_path = './models/merged_outputs/exc_EaSa/model_' + str(args.chkpt) + '.hf'
    tokenizer_path = 't5-small'
    model_dict = load_model(model_name_or_path=model_path, tokenizer_path = tokenizer_path, cuda_devices = [0])
    
    crowdsourced_data = pd.read_csv("./Datasets/annotated_data/annotated_data.csv")
    crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
    textpairs = [[x,y] for x,y in zip(crowdsourced_data['Expert'], crowdsourced_data['Annotation'])]
    
    eval_data = textpairs[-31:]
    print("There are {} eval text pairs".format(len(eval_data)))
    eval_pairs, all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = load_data(eval_data, eval=True)
    
    eval_data = get_eval_data(eval_pairs, slots_eval, all_annotations_eval, in_place_annotation=False)
    batch_instances = batch_for_conditional_gen_merged_outputs(eval_data)
    all_res = run_batch_generation(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'],GENERATOR_OPTIONS_DEFAULT, batch_instances)
    
    #testing_run_macaw(eval_data)
    #batch_instances = create_batch_for_generation(eval_data)
    #eval = run_evaluation(eval_pairs, slots_eval, all_annotations_eval, model_dict)
    
    outputs = []
    true_outputs = []
    inputs = []
    angles = []
    metrics_rouge = []
    metrics_diff = []
    for res in all_res:
        print(res)
        print('\n\n')
        assert len(res['true_output_text'])==len(res['output_slots_list'])
        if len(res['true_output_text'])==len(res['output_slots_list']):
            raw_generated = [v for k, v in res['output_slots_list']]
            rouges = compute_rouge(res['true_output_text'], raw_generated)
            diff = compute_diff(res['true_output_text'], raw_generated)
        else:
            rouges = 0
            diff = -1
        print('text_diff: ', diff)
        metrics_rouge.append(rouges)
        metrics_diff.append(diff)
        inputs.append(res['input'])
        true_outputs.append(res['true_output'])
        outputs.append(res['output_raw_list'])
        angles.append(res['angle'])
        
    #df = pd.DataFrame({'Input':inputs, 'Angle': angles, 'True_outputs':true_outputs, 'Outputs':outputs})
    #df.to_csv('eval_exc_EaSa.csv', index=False)        