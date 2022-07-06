import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np
import re


device = 'cpu'
if torch.cuda.is_available():
    print("using gpu")
    device = 'cuda'

def perplexity(text, tokenizer, lm):
    tokens = tokenizer(text, return_tensors='pt', add_prefix_space=True)
    tokens = tokens.to(device)
    outputs = lm(**tokens, labels=tokens['input_ids'])
    loss = outputs.loss
    if not torch.isnan(loss):
        return loss.mean().item()
    else:
        return 0


def computeppl(texts):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    lm = GPT2LMHeadModel.from_pretrained('gpt2')
    lm = lm.to(device)
    with torch.no_grad():
        score = 0
        for i, text in enumerate(texts):
            score += perplexity(text, tokenizer, lm)
        ppl = torch.exp(torch.tensor(score / i))
    return ppl

def sent_align(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    embd1, embd2 = model.encode(sentences)
    embd1 /= np.linalg.norm(embd1)
    embd2 /= np.linalg.norm(embd2)
    return np.dot(embd1, embd2)

def alignment_matrix(split_text1, split_text2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    embd1 = model.encode(split_text1)
    embd2 = model.encode(split_text2)
    embd1 /= np.linalg.norm(embd1, axis = -1)[:, np.newaxis]
    embd2 /= np.linalg.norm(embd2, axis = -1)[:, np.newaxis]
    dot_matrix = np.dot(embd1, embd2.T)
    return dot_matrix

def max_alignment(dot_matrix):
    return np.max(dot_matrix, axis = 1)

def avg_alignment(text1, text2):
    dot_matrix = alignment_matrix(text1, text2)
    return np.mean(max_alignment(dot_matrix))

def get_sentences(text):
    split_text = []
    spacy_text = spacy_process(text)
    if spacy_text:
        split_text = [s.text for s in spacy_text.sents]
    if split_text and re.search("^\s+$", split_text[-1]):
        del split_text[-1]
    return split_text

def check_elab(split_text1, split_text2, threshold=0.8):
    if len(split_text1) <= len(split_text2):
        return (-1, "-1")
    dot_matrix = alignment_matrix(split_text1, split_text2)
    max_dot = max_alignment(dot_matrix)
    elaborations = []
    for i, x in enumerate(list(max_dot)):
        print(i, " : ", x)
        if x < threshold:
            elaborations.append((i, split_text1[i]))
            #return (i, split_text1[i])
    #return (-1, "-1")
    return elaborations

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

def get_depedency_tree_depth(sentence):
    def subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        else:
            return 1 + max([subtree_depth(child) for child in node.children])
    tree_depths = [subtree_depth(s.root) for s in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    else:
        return max(tree_depths)
    
    
if __name__ == '__main__':
    s = "Hypoglycemia is related to diabetes medication issues. It can also happen if diabetics consume low calories."
    e = "Hypoglycemia is caused by diabetes medication errors or low calorie consumption."
    split_s = get_sentences(s)
    split_e = get_sentences(e)
    print(check_elab(split_s, split_e, 0.7))
