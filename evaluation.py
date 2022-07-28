from collections import defaultdict
import re
import random
from preprocessing import load_data, post_processing_single_angle_only_S
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

GENERATOR_OPTIONS_DEFAULT = {"min_length": 1, "max_length": 128, "num_beams": 10, "num_return_sequences": 1,
                             "do_sample": False, "top_k": 0, "top_p": 0.9, "temperature": 1.0,
                             "length_penalty": 1.0, "repetition_penalty": 1.0}

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
        print(output_string)
        if len(outputs) > 1:
            output_string = ' ; '.join(output_string)
        elif len(output_string) > 0:
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
    return rouges
        
def compute_diff(ar1, ar2):
    seq_mat = difflib.SequenceMatcher()
    diff = []
    ratio = []
    for output, gen in zip(ar1, ar2):
        seq_mat.set_seq1(output)
        seq_mat.set_seq2(gen)
        ratio.append(seq_mat.ratio())
        diff.append(get_replacement(output, gen))
    return diff, ratio

def eval_loop(textpairs, chkpt, model_dict):
    for j in range(0, len(textpairs), 4):
        print('running {} th batch'.format(j))
        test_data = textpairs[j:j+4]
        test_pairs, all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = load_data(test_data, eval=True, single_angle=True)
        all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = post_processing_single_angle_only_S(all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval, out = 'Sa')
        test_data = get_eval_data(test_pairs, slots_eval, all_annotations_eval, in_place_annotation=True)
        batch_instances = batch_for_conditional_gen_merged_outputs(test_data)
        all_res = run_batch_generation(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'],GENERATOR_OPTIONS_DEFAULT, batch_instances)
        outputs = []
        outputs_parsed = []
        true_outputs = []
        true_outputs_parsed = []
        inputs = []
        angles = []
        metrics_rouge = []
        metrics_diff = []
        ratio_metrics_diff = []
        diff_raw_exp = []
        ratio_raw_exp = []
        for res in all_res:
            print('\n\n')
            print(len(res['true_output_text']), len(res['output_slots_list'][0]))
            raw_input = res['input'].split("$expert$ = ")[1].strip()
            if len(res['true_output']) > 0:
                if len(res['true_output_text'])==len(res['output_slots_list'][0]):
                    raw_generated = [v for _, v in res['output_slots_list'][0].items()]
                    print('raw_generated: ', raw_generated)
                    print('true_output: ', res['true_output_text'])
                    rouges = compute_rouge(res['true_output_text'], raw_generated)
                    diff, ratio = compute_diff(res['true_output_text'], raw_generated)
                    exp_diff, exp_ratio = compute_diff([raw_input], [raw_generated[-1]])
                else:
                    print("Some slot is skipped in generation, it is a failure.")
                    rouges = [0] * len(res['true_output_text'])
                    diff = [-1] * len(res['true_output_text'])
                    ratio = [-1] * len(res['true_output_text'])
                    exp_diff = [-1]
                    exp_ratio = [-1]
                print('text_diff: ', (diff, ratio))
                metrics_rouge.append(rouges)
                metrics_diff.append(diff)
                ratio_metrics_diff.append(ratio)
                diff_raw_exp.append(exp_diff)
                ratio_raw_exp.append(exp_ratio[0])
                inputs.append(res['input'])
                true_outputs.append(res['true_output'])
                outputs.append(res['output_raw_list'])
                outputs_parsed.append(raw_generated)
                true_outputs_parsed.append(res['true_output_text'])
                angles.append(res['angle'])
        dir = './results/t5_large/merged_outputs/exc_EaSa_alt_input_format_single_angle/e2sa/dev/'
        df = pd.DataFrame({'Input':inputs, 'Angle': angles, 'True_outputs':true_outputs, 'Outputs':outputs, 'True_outputs_parsed':true_outputs_parsed, 'Outputs_parsed':outputs_parsed, 'Rouge':metrics_rouge, 'Diff_w_true':metrics_diff, 'Diff_w_input':diff_raw_exp, 'Sim_w_true_all':ratio_metrics_diff, 'Sim_w_true': [x[-1] for x in ratio_metrics_diff], 'Sim_w_input':ratio_raw_exp})
        df.to_csv(dir + 'part_files/eval_exc_EaSa_alt_input_format_single_angle_'+str(chkpt)+'_batchno'+str(j)+'.csv', index=False)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type = int, default = 1)
    parser.add_argument('--test', type = int, default = 1)
    parser.add_argument('--batch_size', type  =int, default = 32)
    args = parser.parse_args()
    tokenizer_path = 't5-large'
    crowdsourced_data = pd.read_csv("./Datasets/annotated_data/dev_data.csv", encoding='unicode_escape', engine='python')
    crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
    textpairs = [[x,y,z] for x,y,z in zip(crowdsourced_data['Expert'], crowdsourced_data['Simple'], crowdsourced_data['Annotation'])]
    if args.test == 1:
        model_path = './models/t5_large/merged_outputs/exc_EaSa_alt_input_format_single_angle/e2sa/bs'+str(args.batch_size) +'/model_' + str(args.chkpt) + '.hf'
        model_dict = load_model(model_name_or_path=model_path, tokenizer_path = tokenizer_path, cuda_devices = [0, 1])
        eval_loop(textpairs, args.chkpt, model_dict)
    else:
        for chkpt in range(0, 30):
            model_path = './models/t5_large/merged_outputs/exc_EaSa_alt_input_format_single_angle/e2s/bs8/model_' + str(chkpt) + '.hf'
            model_dict = load_model(model_name_or_path=model_path, tokenizer_path = tokenizer_path, cuda_devices = [0, 1])
            eval_loop(textpairs, chkpt, model_dict)
    
                
