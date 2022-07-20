from model import batch_generator
from collections import defaultdict
import re
import random
from preprocessing import load_data, post_processing_single_angle, get_multiangle_data
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
    

def train_model_with_outputs(model, tokenizer, cuda_device, training_pairs, eval_pairs, shuffle = True, in_place_annotation = True, one_slot = False, data_dir = None, num_epochs=30):
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
                    #print_text(input_string)
                    print('\n\n\t angle for this example: ', angle)
                    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(cuda_device)

                    outputs = []
                    for k, (output_string, output_text) in enumerate(output_strings):
                        output_ids = tokenizer.encode(output_string, return_tensors="pt").to(cuda_device)
                        output_ids[output_ids[:] == tokenizer.pad_token_id] = -100
                        print_text(input_string)
                        print_text(output_string, text_type='Output')
                        res = model(input_ids, labels=output_ids, return_dict=True)
                        res_softmax = torch.softmax(res.logits[0], dim=1)
                        raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
                        output_prob = 1
                        for raw_prob in raw_probs:
                            output_prob *= raw_prob
                        # overall_loss += res.loss
                        #print_text(output_string, text_type='Output')
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

        model.save_pretrained('./models/exc_EaSa_alt_format/model_'+str(epoch)+'.hf')
        
def test_merged_outputs(training_pairs, shuffle = True, in_place_annotation = False, one_slot = False):
    gen = batch_generator(training_pairs, slots_train, all_annotations_train, batch_size = 16, shuffle = shuffle, in_place_annotation = in_place_annotation, one_slot = one_slot)
    for i, batch in enumerate(gen):
        print('batch # {} \n'.format(i))
        for input, output in batch:
            state_dict = get_state_dict(input, output)
            input_string, output_strings, angle = model_input_format(state_dict)
            print('\n\n\t angle for this example: ', angle)
            output_string = [string for string, _ in output_strings]
            output_text = [text for _, text in output_strings]
            if len(output_strings) > 1:
                output_string = ' ; '.join(output_string)
            else:
                output_string = output_string[0]
            print_text(input_string)
            print_text(output_string, text_type='Output')
            #print(output_text)
            
        
        
        
def train_model_with_merged_outputs(model, tokenizer, cuda_device, training_pairs, dev_pairs, num_epochs = 30, batch_size = 4, shuffle = True, in_place_annotation = False, one_slot = False, data_dir = None):
    print("Running model with merged inputs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6)
    for epoch in range(num_epochs):
        epoch_loss = []
        gen = batch_generator(training_pairs, slots_train, all_annotations_train, batch_size = batch_size, shuffle = shuffle, in_place_annotation = in_place_annotation, one_slot = one_slot)
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
                    print('\n\n\t angle for this example: ', angle)
                    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(cuda_device)
                    output_string = [string for string, _ in output_strings]
                    output_text = [text for _, text in output_strings]
                    if len(output_strings) > 1:
                        output_string = ' ; '.join(output_string)
                    else:
                        output_string = output_string[0]
                    output_ids = tokenizer.encode(output_string, return_tensors="pt").to(cuda_device)
                    output_ids[output_ids[:] == tokenizer.pad_token_id] = -100
                    print_text(input_string)
                    print_text(output_string, text_type='Output')
                    res = model(input_ids, labels=output_ids, return_dict=True)
                    res_softmax = torch.softmax(res.logits[0], dim=1)
                    raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
                    output_prob = 1
                    for raw_prob in raw_probs:
                        output_prob *= raw_prob
                    print('\n\t\t loss for output string : {}'.format(res.loss.item()))
                        
                    loss = res.loss / len(batch)
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
                    data_per_batch[i]['res_per_example'].append({'outputs': out, 'input_string': input_string, 'angle': angle})
                data_per_batch[i]['loss_per_batch'] = overall_loss 
                print('\nloss per batch: {}'.format(overall_loss))
                print('\n\n=======================================')
                optimizer.step()
                epoch_loss += [overall_loss]
        average_loss = sum(epoch_loss)/len(epoch_loss)
        print('average loss for epoch # {}: {}'.format(epoch, average_loss))

        eval_res = []
        eval_loss = 0.
        eval_data = get_eval_data(dev_pairs, slots_dev, all_annotations_dev, in_place_annotation=False)
        for input, outputs in eval_data:
            print("\n\n-------------new example------------------")
            res = run_macaw(input, outputs, model_dict=model_dict)
            eval_res.append(res)
            eval_loss += sum([x['loss'] for x in res['explicit_outputs']])/len(res['explicit_outputs'])
        eval_loss /= len(eval_data)
        print('average loss on eval data: {}'.format(eval_loss))
        data_per_batch['eval'] = {'eval_res': eval_res, 'eval_loss': eval_loss}
        
        with open(data_dir + '/result_for_epoch_'+str(epoch)+'.pkl', 'wb') as f:
           pickle.dump(data_per_batch, f)
        
        
        '''PATH = './models/model_'+str(epoch)+'.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            }, PATH)'''

        model.save_pretrained('./models/t5_large/merged_outputs/exc_EaSa_alt_input_format/multiangle/model_'+str(epoch)+'.hf')
        
def print_data(all_inputs, all_outputs, all_annotations, slots):
    for i, val in slots.items():
      print('\nexample #', i)
      print('inputs: ', all_inputs[i])
      print('outputs: ', all_outputs[i])
      #print('annotations: ', all_annotations[i])
      if 'R' in all_annotations[i]:
        print('replacement: ', all_annotations[i]['R'])
      if 'D' in all_annotations[i]:
        print('deletion: ', all_annotations[i]['D'])
      if 'X' in all_annotations[i]:
        print('elaboration: ', all_annotations[i]['X'])
      if 'I' in all_annotations[i]:
        print('insertion: ', all_annotations[i]['I'])
      if 'S' in all_annotations[i]:
        print('simple text: ', all_annotations[i]['S'])
      #print(all_annotations[i]['Sa'])
      for v in val:
        print(slots[i])
        inp, out = v
        print("".join(inp)+'->'+"".join(out))
        

if __name__ == '__main__':
    """_summary_
    args:
        shuffle - training data will be shuffled to create uniform distribution of angles per batch
        ip_ann - in place annotation with tags Ea and Sa, default: False
        one_slot - corresponds to only one angle Ea - > Sa, default: False
        merged_output - slots are all concatenated both in model input and output, default: True
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', type = bool, default = True)
    parser.add_argument('--ip_ann', type = bool, default = False)
    parser.add_argument('--one_slot', type = bool, default = False)
    parser.add_argument('--merged_output', type = bool, default = True)
    parser.add_argument('--multi_angle', type = bool, default = True)
    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--batch_size', type = int, default = 4)
    args = parser.parse_args()
    
    if args.multi_angle:
        train_file = pd.read_csv("./Datasets/annotated_data/processed_train_data.csv")
        training_data = [[x,y,z] for x,y,z in zip(train_file['Expert'], train_file['Simple'], train_file['Annotation'])]
        with open('./Datasets/annotated_data/train_annotations_slots.json', 'r') as f:
            train_dict = json.load(f)
        all_annotations_train = train_dict['annotations']
        slots_train = train_dict['slots']
        slots_train = {int(k):v for k, v in slots_train.items()}
        
        dev_file = pd.read_csv("./Datasets/annotated_data/processed_dev_data.csv")
        dev_data = [[x,y,z] for x,y,z in zip(dev_file['Expert'], dev_file['Simple'], dev_file['Annotation'])]
        with open('./Datasets/annotated_data/dev_annotations_slots.json', 'r') as f:
            dev_dict = json.load(f)
        all_annotations_dev = dev_dict['annotations']
        slots_dev = dev_dict['slots']
        slots_dev = {int(k):v for k, v in slots_dev.items()}
        
        
             
    else:
        train_file = pd.read_csv("./Datasets/annotated_data/train_data.csv", encoding='unicode_escape',engine='python')
        train_file = train_file.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
        training_data = [[x,y,z] for x,y,z in zip(train_file['Expert'], train_file['Simple'], train_file['Annotation'])]
        training_data, all_inputs_train, all_outputs_train, all_annotations_train, slots_train = load_data(training_data, eval=True, single_angle=True)
        #angle_counter_train, all_annotations_train, altered_slots_train = get_multiangle_data(slots_train, all_annotations_train)
        #all_inputs_train, all_outputs_train, all_annotations_train, slots_train = post_processing_single_angle(all_inputs_train, all_outputs_train, all_annotations_train, slots_train, simplify=False)
        #print_data(all_inputs_train, all_outputs_train, all_annotations_train, slots_train)


        dev_file = pd.read_csv("./Datasets/annotated_data/dev_data.csv", encoding='unicode_escape',engine='python')
        dev_file = dev_file.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
        dev_data = [[x,y,z] for x,y,z in zip(dev_file['Expert'], dev_file['Simple'], dev_file['Annotation'])]
        dev_data, all_inputs_dev, all_outputs_dev, all_annotations_dev, slots_dev = load_data(dev_data, eval=True, single_angle=True)
        #angle_counter_eval, all_annotations_eval, altered_slots_eval = get_multiangle_data(slots_eval, all_annotations_eval)
        #all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval = post_processing_single_angle(all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval, simplify=False)
        #print_data(all_inputs_eval, all_outputs_eval, all_annotations_eval, slots_eval)

    print("There are {} training text pairs".format(len(training_data)))
    print("There are {} dev text pairs".format(len(dev_data)))
    
    DEFAULT_RESULTS_DIR = './results/t5_large/merged_outputs/exc_EaSa_alt_input_format/multiangle/'
    model_dict = load_model(model_name_or_path="t5-large", tokenizer_path="t5-large", cuda_devices = [0, 1])

    #test_merged_outputs(training_data)
    
    
    if args.merged_output:
        train_model_with_merged_outputs(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], training_data, dev_data, shuffle = args.shuffle, in_place_annotation = args.ip_ann, one_slot = args.one_slot, data_dir = DEFAULT_RESULTS_DIR)
    else:
        train_model_with_outputs(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'], training_data, dev_data, shuffle = args.shuffle, in_place_annotation = args.ip_ann, one_slot = args.one_slot, data_dir = DEFAULT_RESULTS_DIR)
    

