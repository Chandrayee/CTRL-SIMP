
import strmatching
import dataprep
import argparse
import difflib
import pandas as pd
import spacy
import re
from functools import lru_cache

def punct_space(x):
    for p in ['.', ':', ',', ';', ')']:
        x = x.replace(' '+p, p)
    return x


def replace_with_NA(x):
    if '<elab>' in x:
        return x
    else:
        return 'N.A.'
  
def get_replacement(original_t, rewrit_t):
    a, b = original_t.split(), rewrit_t.split()
    s, r = original_t.lower().split(), rewrit_t.lower().split()
    diff_str = difflib.SequenceMatcher(None, s, r)
    diff_str.a = a
    diff_str.b = b
    diff_anno = strmatching.show_diff(diff_str)
    return diff_anno

def run_rep(expert_text, simple_text):
    annotated = []
    for i, (e, l) in enumerate(zip(expert_text, simple_text)):
        diff_e_l = get_replacement(e, l)
        annotated.append(diff_e_l.strip())
        #strmatching.get_codes(e.lower().split(), l.lower().split())
        print("===========================================\n")
        print(diff_e_l.strip())
    return annotated

@lru_cache(maxsize=1)
def get_spacy_model():
    model = 'en_core_web_sm'
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, model_path=spacy.util.get_path(model))
    return spacy.load(model)

@lru_cache(maxsize=10**6)
def spacy_process(text):
    return get_spacy_model()(text)

def get_sentences(text):
    split_text = []
    spacy_text = spacy_process(text)
    if spacy_text:
        split_text = [s.text for s in spacy_text.sents]
    if split_text and re.search("^\s+$", split_text[-1]):
        del split_text[-1]
    return " ".join(split_text)

def test_spacy(text):
    return get_sentences(text)


print(test_spacy('I am happy.This is good.'))


dir = './Datasets/simpwiki/toloka_pools/general/'
file = 'tim_akina_final_UMLS.csv'

df = pd.read_csv(dir+file, encoding='unicode_escape', engine='python')

keys = df.columns

for y in keys:
    df[y] = df[y].agg(punct_space)
    df[y] = df[y].agg(lambda x: x.strip())
    df[y] = df[y].agg(get_sentences)
    
expert_text = list(df['INPUT:expert'])
simple_text = list(df['INPUT:simple'])
annotated = run_rep(expert_text, simple_text)

df['INPUT:annotation1'] = annotated
    
#df['INPUT:annotation2'] = df['INPUT:annotation2'].agg(replace_with_NA)

print(df.head())

df.to_csv(dir + file, index = False)
