from model import batch_generator
from collections import defaultdict
import re
import random
from preprocessing import load_data
from model import load_model, run_macaw, get_eval_data, model_input_format
import torch
import pandas as pd
import pickle
import json
import numpy as np
from utils import GENERATOR_OPTIONS_DEFAULT
import textwrap
import argparse

tags = {'E->S': 1., 'Ea->Sa': 1., 'E->Sa': 1.}

def get_state_dict(input, output):
    state_dict = {}
    if isinstance(input, str):
        state_dict["input"] = input
    else:
        state_dict["input_fields"] = input
    if isinstance(output, str):
        state_dict["output"] = output
    else:
        state_dict["output_fields"] = output
    return state_dict

def print_text(string, text_type = 'Input'):
    wrapped_string = textwrap.wrap(string)
    print("\n\t" + text_type)
    for line in wrapped_string:
        print('\t ' + line)
    

def train_model_with_outputs(model, tokenizer, cuda_device, training_pairs, eval_pairs, shuffle = True, in_place_annotation = True, one_slot = False, data_dir = None, num_epochs=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6)
    # overall_loss = torch.tensor(0.0, requires_grad=True).to(cuda_device)
    for epoch in range(num_epochs):
        epoch_loss = []
        gen = batch_generator(training_pairs, slots_train, all_annotations_train, batch_size = 16, shuffle = shuffle, in_place_annotation = in_place_annotation, one_slot = one_slot)
        data_per_batch = defaultdict(dict)
        with torch.autograd.set_detect_anomaly(True):
            for i, batch in enumerate(gen):
                print('batch # {} \n'.format(i))
                optimizer.zero_grad() #added here
                overall_loss = 0.
                data_per_batch[i]['res_per_example'] = []
                for input, output in batch:
                    state_dict = get_state_dict(input, output)
                    input_string, output_strings, angle = model_input_format(state_dict)
                    if angle in tags:
                        loss_multiplier = tags[angle]
                    else:
                        loss_multiplier = 1
                    print_text(input_string)
                    print('\n\n\t angle for this example: ', angle)
                    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(cuda_device)

                    outputs = []
                    for k, (output_string, output_text) in enumerate(output_strings):
                        output_ids = tokenizer.encode(output_string, return_tensors="pt").to(cuda_device)
                        res = model(input_ids, labels=output_ids, return_dict=True)
                        res_softmax = torch.softmax(res.logits[0], dim=1)
                        raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
                        output_prob = 1
                        for raw_prob in raw_probs:
                            output_prob *= raw_prob
                        # overall_loss += res.loss
                        print_text(output_string, text_type='Output')
                        print('\n\t\t\t loss for output string {}: {}'.format(k, res.loss.item()))
                        
                        loss = res.loss / len(output_strings) / len(batch)
                        loss *= loss_multiplier
                        out = {'loss': res.loss.item(), 
                        "average loss": loss.item(),
                        'out_string': output_string,
                        'out_text': output_text,
                        "output_tokens": tokenizer.convert_ids_to_tokens(output_ids[0]),
                        "raw_probs": raw_probs,
                        "output_prob": output_prob
                        }
                        loss.backward()
                        overall_loss += loss.item()
                        outputs.append(out)
                    data_per_batch[i]['res_per_example'].append({'outputs': outputs, 'input_string': input_string, 'angle': angle})
                data_per_batch[i]['loss_per_batch'] = overall_loss 
                print('\nloss per batch: {}'.format(overall_loss))
                print('\n\n=======================================')
                optimizer.step()
                epoch_loss += [overall_loss]
        average_loss = sum(epoch_loss)/len(epoch_loss)
        print('average loss for epoch # {}: {}'.format(epoch, average_loss))

        eval_res = []
        eval_loss = 0.
        eval_data = get_eval_data(eval_pairs, slots_eval, all_annotations_eval)
        for input, outputs in eval_data:
            print("\n\n-------------new example------------------")
            print("raw inputs: ", input, '\n')
            res = run_macaw(input, outputs, model_dict=model_dict)
            eval_res.append(res)
            eval_loss += sum([x['loss'] for x in res['explicit_outputs']])/len(res['explicit_outputs'])
        eval_loss /= len(eval_data)
        print('average loss on eval data: {}'.format(eval_loss))
        data_per_batch['eval'] = {'eval_res': eval_res, 'eval_loss': eval_loss}
        
        with open(data_dir + '/result_for_epoch_'+str(epoch)+'.pkl', 'wb') as f:
           pickle.dump(data_per_batch, f)
        
        
        PATH = './models/model_'+str(epoch)+'.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            }, PATH)

        model.save_pretrained('./models/model_'+str(epoch)+'.hf')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', type = bool, default = True)
    parser.add_argument('--ip_ann', type = bool, default = True)
    parser.add_argument('--one_slot', type = bool, default = False)
    args = parser.parse_args()
    
    crowdsourced_data = pd.read_csv("./Datasets/annotated_data/annotated_data.csv")
    crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
    textpairs = [[x,y] for x,y in zip(crowdsourced_data['Expert'], crowdsourced_data['Annotation'])]


    training_data = textpairs[:-31] #-31
    print("There are {} training text pairs".format(len(training_data)))
    training_data, all_inputs_train, all_outputs_train, all_annotations_train, slots_train = load_data(training_data)



    eval_data = textpairs[-31:]
    print("There are {} eval text pairs".format(len(eval_data)))
    eval_data, all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = load_data(eval_data, eval=True)


    DEFAULT_RESULTS_DIR = './results/t5_small'
    model_dict = load_model(model_name_or_path="t5-small", tokenizer_path="t5-small", cuda_devices = [0])

    train_model_with_outputs(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], training_data, eval_data, shuffle = args.shuffle, in_place_annotation = args.ip_ann, one_slot = args.one_slot, data_dir = DEFAULT_RESULTS_DIR)


