import string
import nltk
from nltk import pos_tag
from nltk import tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
import pandas as pd
import random
from collections import defaultdict
import itertools
import numpy as np
from utils import DEFAULT_SLOT_FORMAT, SLOT_SHORTFORMS, ANNOTATION_SLOT_MAP, SLOT_KEY_FROM_LC
import json

"""
This program takes expert text and annotated text and returns all the available 
input and output slots per text pair.
"""

def decompose_slots(input_string, fmt = None):
  fmt = None
  fmt = fmt or DEFAULT_SLOT_FORMAT
  input_string = input_string.strip()
  no_slot = "PREFIX"
  slot_re = re.compile('(?i)'+re.escape(fmt['slot']).replace("SLOT", "(\\w*?)"))
  assign_re = re.escape(fmt['assign']).replace('\\ ','\\s*')
  separator_re = re.escape(fmt['separator']).replace('\\ ','\\s*')
  strip_re = re.compile(f"^({assign_re})?(.*?)({separator_re})?$")
  slot_pos = []
  for m in slot_re.finditer(input_string):
      slot_pos.append((m.span(), m.group(1)))
  if len(slot_pos) == 0:
      return {no_slot: input_string}
  if slot_pos[0][0][0] > 0:
      slot_pos = [((0,-1), no_slot)] + slot_pos
  res = {}
  for idx, (pos, slot_name) in enumerate(slot_pos):
      if idx == len(slot_pos) - 1:
          value = input_string[pos[1]+1:]
      else:
          value = input_string[pos[1]+1:slot_pos[idx+1][0][0]-1]
      m = strip_re.match(value)
      if m is not None:
          value = m.group(2)
      value = value.strip()
      if slot_name in res:
          value = res[slot_name] + " ~AND~ " + value
      res[slot_name] = value
  return res

def normalize_whitespace(input_string):
    return re.sub('\\s+', ' ', input_string).strip()

def decompose_input(string):
    demo_input_regex=re.compile("(?mi)^ *(([A-Za-z]{1,2})(:| *$))? *(.*)")
    res = {}
    key = "NONE"
    for match in demo_input_regex.finditer(string):
        key = match.group(2) or key
        if len(key) == 1:
          key = key.upper()
        else:
          key = key[0].upper() + key[1]
        old = res.get(key,"")
        new = normalize_whitespace(old + " " + match.group(4))
        res[key] = new
    if "C" in res:
        context = res['C']
        del res['C']
        res['C'] = context  # put context in last
    return res

def process_del_punct(s):
  """ removes <del> tags from s, where deleted strings are not NN, JJ, RB, VB, JJS 
      where deleted strings are just punctuations.

  Args:
      s (str): single text with <del> tag

  Returns:
      str, list: edited string and regex match with positions of <del> tags
  """
  start_pattern = '<del>'
  end_pattern = '</' + start_pattern.split('<')[1][:-1] + '>'
  ext_pattern = r'((' + start_pattern + '(.+?)' + end_pattern + '))+'
  all_patterns = []
  for match in re.finditer(ext_pattern, s):
    end = match.end()
    start = match.start()
    all_patterns.append([match.group(3), (start, end)])
  m_string = s
  char_removed = 0
  tags_to_be_removed = []
  for token, dem in all_patterns:
    start, end = dem
    start, end = start - char_removed, end - char_removed
    pos_string = m_string[start + len('<del>') : end - len('</del>')]
    pos_string = pos_string.split()
    tokens_tag = pos_tag(pos_string)
    tags = [x[1] for x in tokens_tag]
    #print('NN present:', 'NN' in tags, 'JJ present:', 'JJ' in tags, 'RB present:', 'RB' in tags, 'VB present:', 'VB' in tags)
    #print('only punctuation deleted: ', re.fullmatch('['+string.punctuation+']+', pos_string[0][0]))
    if (len(tokens_tag) == 1 and 'NN' not in tags and 'JJ' not in tags and 'RB' not in tags and 'VB' not in tags and 'JJS' not in tags) or (len(tokens_tag) == 1 and re.fullmatch('['+string.punctuation+']+', pos_string[0][0])):
      m_string = m_string[:start] + m_string[start + len('<del>'):end - len('</del>')] + m_string[end:]
      tags_to_be_removed.append([token, (start+char_removed, end+char_removed)])
      char_removed += len('<del></del>')
  for x in tags_to_be_removed:
    all_patterns.remove(x)
  return m_string, all_patterns

def process_ins_punct(s):
  """ removes <del> tags from s, where deleted strings are not NN, JJ, RB, VB, JJS 
      where deleted strings are just punctuations.

  Args:
      s (str): single text with <del> tag

  Returns:
      str, list: edited string and regex match with positions of <del> tags
  """
  start_pattern = '<ins>'
  end_pattern = '</' + start_pattern.split('<')[1][:-1] + '>'
  ext_pattern = r'((' + start_pattern + '(.+?)' + end_pattern + '))+'
  all_patterns = []
  for match in re.finditer(ext_pattern, s):
    end = match.end()
    start = match.start()
    all_patterns.append([match.group(3), (start, end)])
  m_string = s
  char_removed = 0
  tags_to_be_removed = []
  for token, dem in all_patterns:
    start, end = dem
    start, end = start - char_removed, end - char_removed
    pos_string = m_string[start + len('<ins>') : end - len('</ins>')]
    pos_string = pos_string.split()
    tokens_tag = pos_tag(pos_string)
    tags = [x[1] for x in tokens_tag]
    #print('NN present:', 'NN' in tags, 'JJ present:', 'JJ' in tags, 'RB present:', 'RB' in tags, 'VB present:', 'VB' in tags)
    #print('only punctuation deleted: ', re.fullmatch('['+string.punctuation+']+', pos_string[0][0]))
    if (len(tokens_tag) == 1 and 'NN' not in tags and 'JJ' not in tags and 'RB' not in tags and 'VB' not in tags and 'JJS' not in tags) or (len(tokens_tag) == 1 and re.fullmatch('['+string.punctuation+']+', pos_string[0][0])):
      m_string = m_string[:start] + m_string[start + len('<ins>'):end - len('</ins>')] + m_string[end:]
      tags_to_be_removed.append([token, (start+char_removed, end+char_removed)])
      char_removed += len('<ins></ins>')
  for x in tags_to_be_removed:
    all_patterns.remove(x)
  return m_string, all_patterns


def process_ins_del(s, pattern):
  """removes <ins> and <del> tags from any input string s

  Args:
      s (str): string containing <ins> or <del> tag
      pattern (str): <ins> or <del>

  Returns:
      str: edited string
  """
  start_pattern = pattern
  end_pattern = '</' + start_pattern.split('<')[1][:-1] + '>'
  ext_pattern = r'((' + start_pattern + '(.+?)' + end_pattern + '))+'
  all_patterns = []
  for match in re.finditer(ext_pattern, s):
    end = match.end()
    start = match.start()
    all_patterns.append((start, end))
  m_string = s
  char_removed = 0
  for start, end in all_patterns:
    start, end = start - char_removed, end - char_removed
    m_string = m_string[:start] + m_string[start + len(start_pattern):end - len(end_pattern)] + m_string[end:]
    char_removed += len(start_pattern+end_pattern)
  return m_string

def make_input_from_example(example):
    slots = decompose_slots(example)
    res_in = []
    res_out = []
    for slot, value in slots.items():
        slot_letter = slot[0].upper()  # Assumes short form is first letter capitalized
        if value.strip() == "":
            res_out.append(slot_letter)
        else:
            res_in.append(f"{slot_letter}: {value}")
    return "\n".join(res_in + res_out)

def make_input_string(fields, angle=None, fmt=None):
    fmt = fmt or DEFAULT_SLOT_FORMAT
    input_slots = []
    input_slots_nice = {}
    output_slots = []
    angle_in = ""
    angle_out = ""
    if angle is None:
        slots = fields
    else:
      #checks the slot with value
        slots = {s:v for s,v in fields.items() if s in angle[0]}
        #updates with empty output slots
        slots.update({s: "" for s in angle[1]})
    for slot, value in slots.items():
        slot_full = SLOT_SHORTFORMS.get(slot, slot).lower()
        slot_short = SLOT_KEY_FROM_LC.get(slot_full, slot_full[0].upper())
        slot_name = fmt['slot'].replace("SLOT", slot_full)
        if value.strip() == "":
          #if there is an empty slot, it means it should be the output
            output_slots.append(slot_name)
            angle_out += slot_short
        else:
          #else if there is value
            input_slots_nice[SLOT_SHORTFORMS.get(slot, slot)] = value
            input_slots.append(f"{slot_name}{fmt['assign']}{value}")
            angle_in += slot_short
    return fmt['separator'].join(output_slots + input_slots), input_slots_nice, angle_in+"->"+angle_out

def extract_pattern(pattern, text):
  """extracts <rep> and <elab> patterns using re

  Args:
      pattern (str): <rep> or <elab>
      text (str): text from which pattern is extracted

  Returns:
      str: re pattern with pattern name, string enclosed, position of patterns
      in text with start and end
  """
  print(text)
  print('\n')
  start_pattern = pattern
  end_pattern = '</' + pattern.split('<')[1][:-1] + '>'
  ext_pattern = r'((' + start_pattern + '(.+?))<by>((.+?)' + end_pattern + '))+'
  pattern_list = []
  for match in re.finditer(ext_pattern,text):
    end = match.end()
    blank_indices = end-len(end_pattern)- len(match.group(5)), end-len(end_pattern)
    pattern_list.append([pattern, match.group(3), match.group(5), match.span(), blank_indices])
  #print('pattern_list: ', pattern_list)
  return pattern_list

def construct_Ea(string, pattern_list):
  """constructs annotated expert text from annotated simple text
  and patterns with positions of <del>, <rep>, <elab> and <elab-sentence> tags

  Args:
      string (str):annotated simple text
      pattern_list (list): contains all patterns present in string including <rep>, 
      <del>, <elab> and <elab-sentence>

  Returns:
      str: annotated expert text
  """
  m_string = string
  char_removed = 0
  
  for p in pattern_list:
    if p[0] == "<elab-sentence>":
      for span in p[-3]:
        start, end = span
        start -= char_removed
        end -= char_removed
        m_string = m_string[:start] + m_string[end:]
        char_removed += end - start
      m_string_tokenized = tokenize.sent_tokenize(m_string)
      for m in m_string_tokenized:
        if '<elab>' not in m and '<rep>' not in m and '<elab-sentence>' not in m and '<elab-define>' not in m:
          m_string_tokenized.remove(m)
      m_string = ''.join(m_string_tokenized)
    elif p[0] == '<del>':
      start, end = p[-2]
      start -= char_removed
      end -= char_removed
      m_string = m_string[:start] + "<del>_</del>" + m_string[end:]
      char_removed += end - start - len("<del>_</del>")
    else:
      start, end = p[-1]
      start -= char_removed
      end -= char_removed
      m_string = m_string[:start] + "_" + m_string[end:]
      char_removed += end - start - 1
      
  return m_string

def construct_S(string, pattern_list):
  """constructs simple text from annotated simple text
  and patterns with positions of <del>, <rep>, <elab> and <elab-sentence> tags

  Args:
      string (str):annotated simple text
      pattern_list (list): contains all patterns present in string including <rep>, 
      <del>, <elab> and <elab-sentence>

  Returns:
      str: simple text with no annotation
  """
  m_string = string
  char_removed = 0
  for p in pattern_list:
    if p[0] == "<elab-sentence>":
      start, end = p[-2]
      start -= len("<elab-sentence>")
      end += len("</elab-sentence>")
      m_string = m_string[:start] + p[-5] + m_string[end:]
      start -= char_removed
      end -= char_removed
      char_removed += end - start - len(p[-5])
      for sentence, span in zip(p[-4], p[-3]):
        start, end = span
        start -= char_removed
        end -= char_removed
        m_string = m_string[:start] + sentence + m_string[end:]
        char_removed += end - start - len(sentence)
    elif p[0] == '<del>':
      start, end = p[-2]
      start -= char_removed
      end -= char_removed
      m_string = m_string[:start] + m_string[end+1:]
      char_removed += end - start + 1
    elif p[0] == '<ins>':
      start, end = p[-2]
      start -= char_removed
      end -= char_removed
      m_string = m_string[:start] + p[-3] + m_string[end:]
      char_removed += len('<ins></ins>')
    else:
      start, end = p[-2]
      start_new, end_new = p[-1]
      start -= char_removed
      end -= char_removed
      start_new -= char_removed
      end_new -= char_removed
      m_string = m_string[:start] + m_string[start_new:end_new] + m_string[end:]
      char_removed += end - start - (end_new - start_new)
  return m_string
    

def get_added_sentence(pattern, text, annotation="token_pos"):
  start_pattern = pattern
  end_pattern = '</' + pattern.split('<')[1][:-1] + '>'
  ext_pattern = r'(' + start_pattern + '(.+?)' + end_pattern + ')+'
  pattern_list = []
  for match in re.finditer(ext_pattern,text):
    start = match.start() + len(start_pattern)
    end = match.end() - len(end_pattern)
    if annotation == "token_pos":
      sentence_marker_start = '<' + str(start) + ',' + str(end) + '>'
      sentence_marker_end = '</' + str(start) + ',' + str(end) + '>'
    else:
      sentence_marker_start = '<' + text[start:end] + '>'
      sentence_marker_end = '</' + text[start:end] + '>'
    sentence_pattern = r'(' + sentence_marker_start + '(.+?)' + sentence_marker_end + ')+'
    matched_sentences = []
    matched_spans = []
    for matched_sentence in re.finditer(sentence_pattern, text):
      matched_sentences.append(matched_sentence.group(2))
      matched_spans.append(matched_sentence.span())
    pattern_list.append([pattern, match.group(2), matched_sentences, matched_spans, (start, end)])

  return pattern_list

def available_slots(texts, alternate_format=True):
  """creates a list of all input and all output slots per text pair

  Args:
      texts (list(str)): a pair of simple and expert texts

  Returns:
      tuple: input slots, output slots and annotation dict eg. {E: text1, S: text2, X: [what is elaborated] ...}
  """
  patterns = ["<rep>", "<elab-define>", "<elab>", "<elab-sentence>", "<del>", "<ins>"]
  annotation_dict = dict()
  if len(texts) == 4:
    e, s, a, c = texts
    input_slots = set(["E", "C"])
    output_slots = set(["S"])
    annotation_dict['C'] = c
  else:
    e, s, a = texts
    input_slots = set("E")
    output_slots = set("S")
  annotation_dict['E'] = e
  annotation_dict['S'] = s
  annotated = False
  #print('\n')
  #print(a)
  all_patterns = []
  for pattern in patterns:
    if pattern != "<elab-sentence>":
      if pattern == "<del>":
        _, pattern_list = process_del_punct(a)
        if len(pattern_list)>0:
          pattern_list = [[pattern, 'None'] + list(x) + [(-1, -1)] for x in pattern_list]
          #print(pattern_list)
          all_patterns += pattern_list
          annotated = True
          input_slots.add(ANNOTATION_SLOT_MAP[pattern])
          output_slots.add(ANNOTATION_SLOT_MAP[pattern])
          annotation_dict[ANNOTATION_SLOT_MAP[pattern]] = [x[2] for x in pattern_list] #span prediction may not be easy, remove all_res[-1]
      elif pattern == "<ins>":
        _, pattern_list = process_ins_punct(a)
        if len(pattern_list)>0:
          pattern_list = [[pattern, 'None'] + list(x) + [(-1, -1)] for x in pattern_list]
          #print(pattern_list)
          all_patterns += pattern_list
          annotated = True
          input_slots.add(ANNOTATION_SLOT_MAP[pattern])
          output_slots.add(ANNOTATION_SLOT_MAP[pattern])
          annotation_dict[ANNOTATION_SLOT_MAP[pattern]] = [x[2] for x in pattern_list]
      else:
        pattern_list = extract_pattern(pattern, a)
        if len(pattern_list)>0:
          #print(pattern_list)
          all_patterns += pattern_list
          annotated = True
          input_slots.add(ANNOTATION_SLOT_MAP[pattern])
          output_slots.add(ANNOTATION_SLOT_MAP[pattern])
          for all_res in pattern_list:
            if not alternate_format:
              if not ANNOTATION_SLOT_MAP[pattern] in annotation_dict:
                annotation_dict[ANNOTATION_SLOT_MAP[pattern]] = [all_res[-4]] #span prediction may not be easy, remove all_res[-1]
              else:
                annotation_dict[ANNOTATION_SLOT_MAP[pattern]].append(all_res[-4])
            else:
              print("Creating alternate data format for elaboration and replacement")
              if not ANNOTATION_SLOT_MAP[pattern] in annotation_dict:
                format = all_res[-4]+' <by> '+all_res[-3]
                annotation_dict[ANNOTATION_SLOT_MAP[pattern]] = [format] #span prediction may not be easy, remove all_res[-1]
              else:
                format = all_res[-4]+' <by> '+all_res[-3]
                annotation_dict[ANNOTATION_SLOT_MAP[pattern]].append(format)
              
            
    else:
      pattern_list = get_added_sentence(pattern, a)
      if len(pattern_list)>0:
        pattern_list = [list(x) + [(-1, -1)] for x in pattern_list]
        all_patterns += pattern_list
        annotated = True
        input_slots.add(ANNOTATION_SLOT_MAP[pattern])
        output_slots.add(ANNOTATION_SLOT_MAP[pattern])
        for p in pattern_list:
          if ANNOTATION_SLOT_MAP[pattern] not in annotation_dict:
            annotation_dict[ANNOTATION_SLOT_MAP[pattern]] = [p[-5]] #span prediction may not be easy, remove p[-1]
          else:
            annotation_dict[ANNOTATION_SLOT_MAP[pattern]].append(p[-5]) #check for number annotation of sentence, is that necessary?

    
  all_patterns = sorted(all_patterns, key = lambda x: x[-2][0])
    

  if annotated:
    input_slots.add("Ea")
    output_slots.add("Sa")
    annotation_dict['Sa'] = a
    #s = construct_S(s, all_patterns)
    #s = process_ins_del(s, '<ins>')
    #annotation_dict['S'] = process_ins_del(s, '<del>')
    ea = construct_Ea(annotation_dict['Sa'], all_patterns)
    annotation_dict['Sa'], _ = process_ins_punct(annotation_dict['Sa'])
    annotation_dict['Sa'], _ = process_del_punct(annotation_dict['Sa'])
    #annotation_dict['Sa'] = process_ins_del(annotation_dict['Sa'] , '<ins>')
    annotation_dict['Ea'] = process_ins_del(ea, '<ins>')

  return input_slots, output_slots, annotation_dict

def get_singleangle_data(complete_input_slots, complete_output_slots, complete_annotation_dict):
  data = []
  for x, y, z in zip(complete_input_slots, complete_output_slots, complete_annotation_dict):
    data.append((x, y, z))
  return data
    


def get_balanced_data(complete_input_slots, complete_output_slots, complete_annotation_dict, eval=False):
    '''
    Identify the pairs with 'X' and 'D' slots. Decompose them to make more samples
    Add the new samples with previous data
    Separate out the angles and sample from them uniformly to get the balanced text pairs
    '''
    
    slot_list = ['X', 'R', 'D', 'I']
    
    raw_data = defaultdict(list)
    data_for_resample = defaultdict(list)
    
    for slot in slot_list:
      for idx,slots in enumerate(complete_input_slots):
          if slot in slots:
              raw_data[slot].append((complete_input_slots[idx], complete_output_slots[idx], complete_annotation_dict[idx]))
              input_slots = {'E', slot}
              output_slots = {'S', slot}
              annotation_dict = {'E': complete_annotation_dict[idx]['E'], 'S': complete_annotation_dict[idx]['S'], 
                                       slot: complete_annotation_dict[idx][slot]}
              data_for_resample[slot].append((input_slots, output_slots, annotation_dict))  
            

    print('R: {}, D: {}, X: {}, I: {}'.format(len(raw_data['R']), len(raw_data['D']), len(raw_data['X']), len(raw_data['I'])))
    
    resampled_data = list(itertools.chain.from_iterable([raw_data[key] for key in raw_data.keys()]))
    letter_counts = {x: len(y) for x, y in zip(slot_list, raw_data.values())}
    max_count = max(letter_counts.values())
    
    print('sampled data: ', len(resampled_data))
    # change the proportion or desired distribtuion of labels here
    
    if not eval:
      for slot in slot_list:
        resampled_data = resample(resampled_data, slot, raw_data, data_for_resample, max_count)
    
    
    print('resampled data: ', len(resampled_data))
    
    return resampled_data
  
def resample(resampled_data, slot, raw_data, data_for_resample, max_count, repetition=False):
  proportion = {'R': 1., 'X': 1.,'D': 1., 'XS': 1., 'I': 1}
  if len(raw_data[slot]) > 0:
    count = int(proportion[slot] * (max_count - len(raw_data[slot])))
    if repetition:
        i = 0
        while i < count:
          resampled_data += random.sample(data_for_resample[slot], 1)
          i += 1
        return resampled_data
    while count > 0:
      resampled_data += data_for_resample[slot][:count]
      count -= len(data_for_resample[slot])
  return resampled_data
  
def get_annotations(textpairs):
  all_inputs = []
  all_outputs = []
  all_annotation = []
  for i, texts in enumerate(textpairs):
    print(i)
    input_slots, output_slots, annotation_dict = available_slots(texts)
    all_inputs.append(input_slots)
    all_outputs.append(output_slots)
    all_annotation.append(annotation_dict)
  return all_inputs, all_outputs, all_annotation
  
def prep_final_slots_per_datapoint(textpairs, sampling = False):
  count_primary = 0
  primary_slots = []
  all_inputs = []
  all_outputs = []
  all_annotations = []
  slots = defaultdict(list)
  for i, (inputs, outputs, annotations) in enumerate(textpairs):
    slots[i].append((['E'],['S']))
    control_slots = inputs.intersection(set(['X','C','R','F','Xs', 'D', 'I']))
    slots[i].append((['E']+ list(control_slots),['S']))
    slots[i].append((['E'],list(control_slots) + ['S']))
    if 'Sa' in outputs:
      slots[i].append((['E'],['Sa']))
      slots[i].append((['E']+ list(control_slots),['Sa']))
      slots[i].append((['E'],list(control_slots) + ['Sa']))
    all_inputs.append(inputs)
    all_outputs.append(outputs)
    all_annotations.append(annotations)
    primary_slots.append(i)
    count_primary += 1
    if not sampling and 'Ea' in annotations:
      slots[i].append((['Ea'],['Sa']))
      slots[i].append((['Ea'],['S']))
  if not sampling:
    return all_inputs, all_outputs, all_annotations, slots
  
  #we can also use sampling to down sample easier examples like Ea to Sa per batch
  fraction_Ea_Sa = int(np.ceil(0.7 * count_primary)) #0.4
  fraction_Ea_S = int(np.ceil(0.2 * count_primary)) #0.2
  fraction_E_S = int(np.ceil(0.8 * count_primary)) #0.4

  Ea_Sa = np.random.choice(primary_slots, fraction_Ea_Sa)
  Ea_S = np.random.choice(primary_slots, fraction_Ea_S)
  E_S = np.random.choice(primary_slots, fraction_E_S)

  for i in Ea_Sa:
    slots[i].append((['Ea'],['Sa']))

  for i in Ea_S:
    slots[i].append((['Ea'],['S']))

  for i in E_S:
    slots[i].append((['E'],['S']))
    control_slots = all_inputs[i].intersection(set(['X','C','R','F','Xs','I']))
    slots[i].append((['E']+ list(control_slots),['S']))
    slots[i].append((['E'],['S'] + list(control_slots)))

  return all_inputs, all_outputs, all_annotations, slots

def get_training_data(annotation_dict, slot, in_place_annotation = True):
  inputs = []
  outputs = []
  print(annotation_dict)
  print(slot)
  for inp, out in slot:
    if not in_place_annotation:
      if 'Ea' in inp:
        inp = inp.remove('Ea')
      if 'Sa' in out:
        out = out.remove('Sa')
    if inp and out:
      inpcopy = inp.copy()
      output_strings = []
      if 'Ea' in inpcopy and 'Ea' in annotation_dict:
        inp = inpcopy.remove('Ea')
        input_string = 'Ea:' + annotation_dict['Ea']
      elif 'E' in inp:
        inp = inpcopy.remove('E')
        input_string = 'E:' + annotation_dict['E']
      for more_slot in inpcopy:
        if more_slot in annotation_dict:
          text = " | ".join([str(x) for x in annotation_dict[more_slot]])
          text = '[' + text + ']'
          #input_string += "\n" + more_slot + ':' + str(annotation_dict[more_slot]) 
          input_string += "\n" + more_slot + ':' + text
      input_string += "\n"+"\n".join(out)
      for output_slot in out:
        if output_slot in annotation_dict:
          if output_slot == 'S' or output_slot == 'Sa':
            output_strings.append(output_slot + ":" + str(annotation_dict[output_slot]))
            #output_string = output_slot + ":" + str(annotation_dict[output_slot])
          else:
            text = " | ".join([str(x) for x in annotation_dict[output_slot]])
            text = '[' + text + ']'
            output_strings.append(output_slot + ':' + text)
      print(input_string)
      print(output_strings)
      inputs.append(input_string)
      outputs.append(output_strings)

  return inputs, outputs

def get_inplace_annotated_data(annotation_dict, slot):
  inputs = []
  outputs = []
  for inp, out in slot:
    if 'Ea' in inp and 'Ea' in annotation_dict and 'Sa' in out and 'Sa' in annotation_dict:
      input_string = 'Ea:' + annotation_dict['Ea']
      input_string += "\n"+"\n".join(out)
      output_string = 'Sa:' + annotation_dict['Sa']
      print(input_string)
      print(output_string)
      print("===============\n")
      inputs.append(input_string)
      outputs.append([output_string])
  return inputs, outputs
  
def load_data(textpairs, eval = False, single_angle = False):
  complete_input_slots, complete_output_slots, complete_annotation_dict = get_annotations(textpairs)
  if single_angle:
    textpairs = get_singleangle_data(complete_input_slots, complete_output_slots, complete_annotation_dict)
  else:
    textpairs = get_balanced_data(complete_input_slots, complete_output_slots, complete_annotation_dict, eval=eval)
  all_inputs, all_outputs, all_annotations, slots = prep_final_slots_per_datapoint(textpairs)
  return textpairs, all_inputs, all_outputs, all_annotations, slots

def post_processing_single_angle(all_inputs, all_outputs, all_annotations, slots, simplify=True):
  for i, val in slots.items():
    all_inputs[i] = {'E'}
    all_outputs[i] = {'R', 'X', 'D', 'I'}
    all_keys = ['E', 'R', 'X', 'D', 'I']
    slots[i] = [(['E'], ['R', 'X', 'D', 'I'])]
    if simplify:
      all_outputs[i].add('S')
      all_keys += ['S']
      slots[i] = [(['E'], ['R', 'X', 'D', 'I', 'S'])]
    for key in all_keys:
      if key not in all_annotations[i]:
        all_annotations[i][key] = ['<extra_id_0>']
  return all_inputs, all_outputs, all_annotations, slots

def post_processing_single_angle_only_S(all_inputs, all_outputs, all_annotations, slots):
  for i, val in slots.items():
    all_inputs[i] = {'E'}
    all_outputs[i] = {'S'}
    slots[i] = [(['E'], ['S'])]
  return all_inputs, all_outputs, all_annotations, slots

def test_eval(textpairs, annotation_dict, slots):
  batch = []
  for i in range(len(textpairs)):
    inputs, outputs = get_training_data(annotation_dict[i], slots[i], in_place_annotation = False)
    for input, output in zip(inputs, outputs):
      #per angle for one example
      batch.append((input, output))
  return batch

def multiangle_process(slots):
  slot_types = ['R', 'X']
  for slot in slot_types:
      angleset = set()
      save_slot = []
      remove_slot = []
      for val in slots:
          rand_val = random.random()
          #print('random: ', rand_val)
          inp, out = val
          #print('val:', val)
          #print('angleset: ', angleset)
          if slot in angleset:
              #print('yes')
              #print(inp, out)
              if slot in inp:
                  inp.remove(slot)
              if slot in out:
                  out.remove(slot)
              #print([inp, out])
          if slot in inp and slot not in angleset and slot not in out:
              out += slot
              angleset.add(slot)
          elif slot in out and slot not in angleset and slot not in inp:
              if rand_val>0.5:
                  inp += slot
              angleset.add(slot)
          if val != [inp, out]:
              remove_slot.append(val)
              save_slot.append([inp, out])
              
      for x in list(remove_slot):
          if x in slots:
              slots.remove(x)
      
      for x in list(save_slot):
          if x not in slots:
              slots.append(x)
              
      #print(slots)
      
      
  save_slot = []
  remove_slot = []
  d_inp = False
  d_out = False
  for val in slots:
      rand_val = random.random()
      #print('random: ', rand_val)
      inp, out = val
      #print('val:', val)
      if 'D' in inp:
        if rand_val > 0.5:
          d_inp = True
        else:
          inp.remove('D')
          out = ['D'] + out
          d_out = True
      if 'D' in inp and d_out:
        inp.remove('D')
        out = ['D'] + out
      elif 'D' in out and d_inp:
        out.remove('D')
        inp += ['D']
      
      save_slot.append([inp, out])
      remove_slot.append(val)
      
  for x in list(remove_slot):
          if x in slots:
              slots.remove(x)
      
  for x in list(save_slot):
      if x not in slots:
          slots.append(x)
          
          
  save_slot = []
  remove_slot = []        
          
  for val in slots:
      inp, out = val
      if 'I' in inp:
          inp.remove('I')
          remove_slot.append(val)
          save_slot.append([inp, out])
        
  for x in list(remove_slot):
          if x in slots:
              slots.remove(x)
      
  for x in list(save_slot):
      if x not in slots:
          slots.append(x)
                  
  return slots


def get_multiangle_data(slots, all_annotations):
  angle_counter = defaultdict(int)
  altered_slots = defaultdict(list)
  for i, val in slots.items():
    val = multiangle_process(val)
    for v in val:
      inp, out = v
      if not 'Ea' in inp and not 'Sa' in out:
        inp[1:] = sorted(inp[1:])
        out.remove('S')
        out.append('S')
        out[:-1] = sorted(out[:-1])
        if 'R' in inp:
          idx = inp.index('R')
          inp[idx] = 'Ri'
          if 'Ri' not in all_annotations[i]:
            all_annotations[i]['Ri'] = [x.split(' <by> ')[0] for x in all_annotations[i]['R']]
        if 'X' in inp:
          idx = inp.index('X')
          inp[idx] = 'Xi'
          if 'Xi' not in all_annotations[i]:
            all_annotations[i]['Xi'] = [x.split(' <by> ')[0] for x in all_annotations[i]['X']]
        angle = ''.join(inp) + '->' + ''.join(out)
        angle_counter[angle] += 1
        altered_slots[i].append([inp, out])
        
        
  return angle_counter, all_annotations, altered_slots
  

  
    

if __name__ == '__main__':
    
    # read the annotated data
    # remove the duplicated entries
    #take fraction of text data for testing
    crowdsourced_data = pd.read_csv("./Datasets/annotated_data/eval_data.csv", encoding='unicode_escape', engine='python')
    crowdsourced_data = crowdsourced_data.drop_duplicates( subset = ['Expert', 'Simple'], keep = 'last').reset_index(drop = True)
    textpairs = [[x,y,z] for x,y,z in zip(crowdsourced_data['Expert'], crowdsourced_data['Simple'], crowdsourced_data['Annotation'])]
    #textpairs = textpairs[:2]
    textpairs, all_inputs, all_outputs, all_annotations, slots = load_data(textpairs, eval=True, single_angle=True)
    #print(slots[:2])
    all_inputs, all_outputs, all_annotations, slots = post_processing_single_angle_only_S(all_inputs, all_outputs, all_annotations, slots)
    #angle_counter, all_annotations, altered_slots = get_multiangle_data(slots, all_annotations)
    #print(altered_slots)

    print('There are {} eval data'.format(len(textpairs)))
    #print(angle_counter)
    
    batch = test_eval(textpairs, all_annotations, slots)
    
    for input, outputs in batch:
      print('Input')
      print(input, '\n')
      print('Labels')
      print(outputs, '\n')

    '''
  
    crowdsourced_data.to_csv('processed_eval_data.csv', index = False)
    
    with open('eval_annotations_slots.json', 'w') as f:
      json.dump({'angle_counter': angle_counter, 'annotations': all_annotations, 'slots': altered_slots}, f)
      
    ''' 
