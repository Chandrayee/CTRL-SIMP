from model import batch_generator
from collections import defaultdict
import re
import random
from preprocessing import load_data
from model import load_model, run_model, run_model_with_outputs, get_eval_data
import torch
import pandas as pd
import pickle
import json
import numpy as np
from utils import GENERATOR_OPTIONS_DEFAULT
import textwrap

tags = {'Sa': 1., 'X': 1., 'S': 1., 'D': 1., 'X': 1.}


def train_model_with_outputs(model, tokenizer, cuda_device, training_pairs, eval_pairs, data_dir = None, num_epochs=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6)
    # overall_loss = torch.tensor(0.0, requires_grad=True).to(cuda_device)
    for epoch in range(num_epochs):
        epoch_loss = []
        gen = batch_generator(training_pairs, slots_train, all_annotations_train, batch_size = 16)
        data_per_batch = defaultdict(dict)
        with torch.autograd.set_detect_anomaly(True):
            for i, batch in enumerate(gen):
                print('batch # {} \n'.format(i))
                optimizer.zero_grad() #added here
                overall_loss = 0.
                data_per_batch[i]['res_per_example'] = []
                for input_string, output_strings in batch:
                    wrapped_input = textwrap.wrap(input_string)
                    print("\n\t Input: ")
                    for line in wrapped_input:
                        print('\t ' + line)
                    input_string = input_string
                    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(cuda_device)

                    outputs = []
                    for k, output_string in enumerate(output_strings):
                        for tag, val in tags.items():
                            if output_string.startswith(tag+':'):
                                multiplier = val
                                final_tag = tag
                                break
                        print('\n\t\tloss multiplier for {}: {}'.format(final_tag, multiplier))
                        output_ids = tokenizer.encode(output_string, return_tensors="pt").to(cuda_device)
                        res = model(input_ids, labels=output_ids, return_dict=True)
                        res_softmax = torch.softmax(res.logits[0], dim=1)
                        raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
                        output_prob = 1
                        for raw_prob in raw_probs:
                            output_prob *= raw_prob
                        # overall_loss += res.loss
                        print('\t\t output string {}:'.format(k))
                        wrapped_output = textwrap.wrap(output_string)
                        for line in wrapped_output:
                            print('\t\t '+line)
                        print('\t\t loss for output string {}: {}'.format(k, res.loss.item()))
                        
                        loss = res.loss / len(output_strings) / len(batch)
                        loss *= multiplier
                        out = {'loss': res.loss.item(), 
                        "average loss": loss.item(),
                        'out_string': output_string,
                        "output_tokens": tokenizer.convert_ids_to_tokens(output_ids[0]),
                        "raw_probs": raw_probs,
                        "output_prob": output_prob
                        }
                        loss.backward()
                        overall_loss += loss.item()
                        outputs.append(out)
                    data_per_batch[i]['res_per_example'].append({'outputs': outputs, 'input_string': input_string})
                data_per_batch[i]['loss_per_batch'] = overall_loss 
                print('\nloss per batch: {}'.format(overall_loss))
                print('\n\n=======================================')
                optimizer.step()
                epoch_loss += [overall_loss]
        average_loss = sum(epoch_loss)/len(epoch_loss)
        print('average loss for epoch # {}: {}'.format(epoch, average_loss))

        eval_res = []
        eval_gen = []
        eval_loss = 0.
        eval_data = get_eval_data(eval_pairs, slots_eval, all_annotations_eval)
        for _, (input_string, output_strings) in enumerate(eval_data):
            res, loss = run_model_with_outputs(model, tokenizer, cuda_device, input_string, output_strings)
            res_gen = run_model(model, tokenizer, cuda_device, input_string, GENERATOR_OPTIONS_DEFAULT)
            eval_res.append(res)
            eval_gen.append(res_gen)
            eval_loss += loss
        eval_loss /= len(eval_data)
        print('average loss on eval data: {}'.format(eval_loss))
        data_per_batch['eval'] = {'eval_res': eval_res, 'eval_gen': eval_gen, 'eval_loss': eval_loss}
        
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
        


crowdsourced_data = pd.read_csv("./Datasets/annotated_data/annotated_data.csv")
crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
textpairs = [[x,y] for x,y in zip(crowdsourced_data['Expert'], crowdsourced_data['Annotation'])]


training_data = textpairs[:-31] #-31
print("There are {} training text pairs".format(len(training_data)))
training_data, all_inputs_train, all_outputs_train, all_annotations_train, slots_train = load_data(training_data)



eval_data = textpairs[-31:]
print("There are {} eval text pairs".format(len(eval_data)))
eval_data, all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = load_data(eval_data)


DEFAULT_RESULTS_DIR = './results'
model_dict = load_model(model_name_or_path="t5-small", cuda_devices = [0])

train_model_with_outputs(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], training_data, eval_data, data_dir = DEFAULT_RESULTS_DIR)


