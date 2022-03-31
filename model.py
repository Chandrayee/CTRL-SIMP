from preprocessing import decompose_input, make_input_string, decompose_slots, get_training_data, get_inplace_annotated_data
from utils import DEFAULT_SLOT_FORMAT, SLOT_SHORTFORMS, GENERATOR_OPTIONS_DEFAULT
from collections import defaultdict
import re
import random
import torch
import math
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import itertools
    
def load_model(model_name_or_path, tokenizer_path, cuda_devices=None):
  #suppose there are 13 layers
  #one_extra makes sure that the last layer gets added to the last gpu
  cuda_devices = cuda_devices or []
  tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
  model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
  device_map = None
  if len(cuda_devices) > 1:
    num_layers = model.config.num_layers
    n_gpu = len(cuda_devices)
    layers_per_gpu = num_layers // n_gpu
    has_one_extra = n_gpu - (num_layers - layers_per_gpu*n_gpu)
    device_map = {}
    current = 0
    for device in cuda_devices:
      next = current + layers_per_gpu
      if len(device_map) >= has_one_extra:
        #adding the last layer to the last gpu
        next += 1
      device_map[device] = list(range(current, next))
      current = next
  if len(cuda_devices) > 0:
    device = f"cuda:{cuda_devices[0]}"
  else:
    device = "cpu"

  if device_map is not None:
    model.parallelize(device_map)
  else:
    model.to(device)
  
  return {'model': model, 'tokenizer': tokenizer, 'cuda_device': device}

def run_model(model, tokenizer, cuda_device, input_string, generator_options):
  with torch.no_grad():
    input_string = input_string
    input_ids = tokenizer.encode(input_string, return_tensors = 'pt').to(cuda_device)
    res = model.generate(input_ids, **generator_options)
    output_strings = tokenizer.batch_decode(res, skip_special_tokens = True)
    res = {"input_raw": input_string, "output_raw_list": output_strings}
    print("generated texts: ", res)

  return res

def compute_answer(model, tokenizer, cuda_device, raw_input, generator_options, output_strings=None):
  res = run_model(model, tokenizer, cuda_device, raw_input, generator_options)
  res['generator_options'] = generator_options
  if output_strings is not None:
    res['explicit_outputs'], _ = run_model_with_outputs(model, tokenizer, cuda_device, raw_input, output_strings)
    #print(res['explicit_outputs'])
    res['explicit_outputs'].sort(key=lambda x: -x['score'])
  return res

def model_input_format(state_dict):
  input_string = state_dict['input']
  input_fields = decompose_input(input_string)
  explicit_outputs = state_dict['output_fields']
  raw_input, input_slots, angle = make_input_string(input_fields)
  ouput_strings = None
  angle_out = angle.split("->")[1]
  if explicit_outputs is not None:
    explicit_outputs = [decompose_input(output) for output in explicit_outputs]
    texts = [[v for _, v in d.items()] for d in explicit_outputs]
    output_strings = [(make_input_string(output)[0], t[0]) for output, t in zip(explicit_outputs, texts)]
    
  return raw_input, output_strings, angle
  
    
def get_raw_response(state_dict, compute_answer_fn=None, model_dict=None):
  input_string = state_dict['input']
  input_fields = decompose_input(input_string)
  explicit_outputs = state_dict['output_fields']
  raw_input, input_slots, angle = make_input_string(input_fields)
  ouput_strings = None
  angle_out = angle.split("->")[1]
  if explicit_outputs is not None:
    explicit_outputs = [decompose_input(output) for output in explicit_outputs]
    texts = [[v for _, v in d.items()] for d in explicit_outputs]
    output_strings = [(make_input_string(output)[0], t[0]) for output, t in zip(explicit_outputs, texts)]
   


  generator_options = GENERATOR_OPTIONS_DEFAULT.copy()
  for key, init_val in generator_options.items():
      if key in state_dict:
          val = state_dict[key]
          if isinstance(val, str):
              if isinstance(init_val, bool):
                  val = int(val) > 0
              elif isinstance(init_val, int):
                  val = int(val)
              else:
                  val = float(val)
          generator_options[key] = val

  if compute_answer_fn is not None:
        random_tickle = 1
        # Trick to make sure streamlit doesn't cache responses if sampling is used in generator settings
        if generator_options['do_sample']:
            random_tickle = random.random()
        res_raw = compute_answer_fn(raw_input, generator_options, random_tickle, output_strings)
  else:
      assert model_dict is not None
      res_raw = compute_answer(model_dict['model'], model_dict['tokenizer'], model_dict['cuda_device'],
                                raw_input, generator_options, output_strings)
  res = res_raw.copy()
  res['requested_angle'] = angle
  res['input_slots'] = input_slots
  res['output_slots_list'] = [decompose_slots(output) for output in res['output_raw_list']]
  if explicit_outputs is not None:
      res['explicit_output_angle'] = SLOT_SHORTFORMS.get(angle_out, angle_out)
  return res

def run_model_with_outputs(model, tokenizer, cuda_device, input_string, output_strings):
  with torch.no_grad():
    input_string = input_string
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(cuda_device)
    all_res = []
    average_loss = 0.
    for output_string, output_text in output_strings:
      output_ids = tokenizer.encode(output_string, return_tensors="pt").to(cuda_device)
      res = model(input_ids, labels=output_ids, return_dict=True)
      res_softmax = torch.softmax(res.logits[0], dim=1)
      #what is going on here?
      raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
      output_prob = 1
      for raw_prob in raw_probs:
        output_prob *= raw_prob
      loss = res.loss.item()
      average_loss += loss
      all_res.append({
        "input_raw": input_string,
        "output_raw": output_string,
        "output_text": output_text,
        "loss": loss,
        "score": math.exp(-loss),
        "output_prob": output_prob,
        "output_token_probs": raw_probs,
        "output_tokens": tokenizer.convert_ids_to_tokens(output_ids[0])
      })
    average_loss /= len(output_strings)
  return all_res, average_loss
    #weighted sum(loss) weighted by text length
    
def shuffle_order(x):
  random_state = 100
  np.random.seed(random_state)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = [x[ind] for ind in indices]
  return x
  

def batch_generator(textpairs, slots, all_annotation, batch_size=16, shuffle = True, in_place_annotation = True, one_slot = False):
  batch = []
  all_inputs = []
  all_outputs = []
    
  for i in range(len(textpairs)):
    if one_slot:
      print("Running only Ea->Sa --------\n")
      inputs, outputs = get_inplace_annotated_data(all_annotation[i], slots[i])
    else:
      print("Running all slots --------\n")
      inputs, outputs = get_training_data(all_annotation[i], slots[i], in_place_annotation = in_place_annotation) 
    
    for input, output in zip(inputs, outputs):
      all_inputs.append(input)
      all_outputs.append(output)
    #option 1: treating every example equally - keep a max combination, repeat example that do not have enough combination
    #higher weight to harder example, each batch should be diverse (cos similarity)
  
  if shuffle:
    print("Examples are shuffled to create heterogeneous batch --------\n")
    all_inputs = shuffle_order(all_inputs)
    all_outputs = shuffle_order(all_outputs)
      
    
  for input, output in zip(all_inputs, all_outputs):
    #per angle for one example
    batch.append((input, output))
    if len(batch) == batch_size:
      yield batch
      batch = []
      
  if batch:
    yield batch

def get_eval_data(textpairs, slots, all_annotation, shuffle = False, in_place_annotation = True, one_slot = False):
  batch = []  
  for i in range(len(textpairs)):
    if one_slot:
      print("Running only Ea->Sa --------\n")
      inputs, outputs = get_inplace_annotated_data(all_annotation[i], slots[i])
    else:
      print("Running all slots --------\n")
      inputs, outputs = get_training_data(all_annotation[i], slots[i], in_place_annotation = in_place_annotation) 
    #option 1: treating every example equally - keep a max combination, repeat example that do not have enough combination
    #higher weight to harder example, each batch should be diverse (cos similarity)
    for input, output in zip(inputs, outputs):
      #per angle for one example
      batch.append((input, output))
  
  if shuffle:
    batch = shuffle_order(batch)

  return batch


def run_macaw(input, output, model_dict, generator_options=None):
  state_dict = {}
  if isinstance(input, str):
    state_dict["input"] = input
  else:
    state_dict["input_fields"] = input

  if isinstance(output, str):
    state_dict["output"] = output
  else:
    state_dict["output_fields"] = output

  if generator_options is not None:
    state_dict.update(generator_options)

  res = get_raw_response(state_dict, model_dict=model_dict)
  return res



