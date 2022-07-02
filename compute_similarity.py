import pandas as pd
import metrics


import Levenshtein
import nltk
from functools import lru_cache
import spacy
nltk.download('punkt')
from collections import Counter
import textstat
import numpy as np
from collections import Counter

#codes for sentence-pair analysis
#all codes except for bleu and compression_corpus taken from https://github.com/facebookresearch/text-simplification-evaluation/blob/c86046930a2144406d941a99127eba795eea6296/tseval/feature_extraction.py

def safe_division(a, b):
    if b == 0:
        return b
    return a / b

def count_words(sentence, tokenize=True, remove_punctuation=False):
    if tokenize:
        sentence = word_tokenize(sentence)
    if remove_punctuation:
        sentence = remove_punctuation_tokens(sentence)
    return len(to_words(sentence))

def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')


def wrap_single_sentence_vectorizer(vectorizer):
    '''Transform a single sentence vectorizer to a sentence pair vectorizer
    Change the signature of the input vectorizer
    Initial signature: method(simple_sentence)
    New signature: method(complex_sentence, simple_sentence)
    '''
    def wrapped(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence)

    wrapped.__name__ = vectorizer.__name__
    return wrapped

@lru_cache(maxsize=1)
def get_nist_tokenizer():
    # Inline lazy import because importing nltk is slow
    try:
        from nltk.tokenize.nist import NISTTokenizer
    except LookupError:
        import nltk
        nltk.download('perluniprops')
    return NISTTokenizer()

@lru_cache(maxsize=100)  # To speed up subsequent calls
def word_tokenize(sentence):
    return ' '.join(get_nist_tokenizer().tokenize(sentence))

def to_words(sentence):
    return sentence.split()

def count_words_per_sentence(sentence):
    return safe_division(count_words(sentence), count_sentences(sentence))

def to_sentences(text, language='english'):
    # Inline lazy import because importing nltk is slow
    tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    return tokenizer.tokenize(text)

def count_sentences(text, language='english'):
    return len(to_sentences(text, language))

def count_characters(sentence):
    return len(sentence)

def count_characters_per_sentence(sentence):
    return safe_division(count_characters(sentence), count_sentences(sentence))

def count_sentence_splits(complex_sentence, simple_sentence):
    return safe_division(count_sentences(simple_sentence), count_sentences(complex_sentence))


def get_compression_ratio(complex_sentence, simple_sentence):
    return safe_division(count_characters(simple_sentence), count_characters(complex_sentence))

def characters_per_sentence_difference(complex_sentence, simple_sentence):
    return count_characters_per_sentence(complex_sentence) - count_characters_per_sentence(simple_sentence)

def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)

def get_levenshtein_distance(complex_sentence, simple_sentence):
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)

def get_additions_proportion(complex_sentence, simple_sentence):
    n_additions = sum((Counter(to_words(simple_sentence)) - Counter(to_words(complex_sentence))).values())
    return n_additions / max(count_words(complex_sentence), count_words(simple_sentence))

def get_deletions_proportion(complex_sentence, simple_sentence):
    n_deletions = sum((Counter(to_words(complex_sentence)) - Counter(to_words(simple_sentence))).values())
    return n_deletions / max(count_words(complex_sentence), count_words(simple_sentence))

def flatten_counter(counter):
    return [k for key, count in counter.items() for k in [key] * count]


def get_added_words(c, s):
    return flatten_counter(Counter(to_words(s)) - Counter(to_words(c)))


def get_deleted_words(c, s):
    return flatten_counter(Counter(to_words(c)) - Counter(to_words(s)))


def get_kept_words(c, s):
    return flatten_counter(Counter(to_words(c)) & Counter(to_words(s)))

def get_n_added_words(c, s):
    return len(get_added_words(c, s))


def get_n_deleted_words(c, s):
    return len(get_deleted_words(c, s))


def get_n_kept_words(c, s):
    return len(get_kept_words(c, s))

def max_characters_per_sentence(text):
  max_len = 0
  for sentence in to_sentences(text):
    n_chars = count_characters(sentence)
    max_len = max(max_len, n_chars)
  return max_len

def avg_characters_per_sentence(text):
  avg_len = 0
  for sentence in to_sentences(text):
    n_chars = count_characters(sentence)
    avg_len += n_chars
  avg_len /= count_sentences(text)
  return avg_len

def compression_corpus(c, s, tag = 'max'):
  if tag == 'max':
    layman, expert = max_characters_per_sentence(s), max_characters_per_sentence(c)
  elif tag == 'avg':
    layman, expert = avg_characters_per_sentence(s), avg_characters_per_sentence(c)
  return safe_division(layman, expert)

def get_bleu(reference, candidate):
  from nltk.translate.bleu_score import sentence_bleu
  score = sentence_bleu(reference, candidate)
  return score

def get_corpus_bleu(references, candidates):
  from nltk.translate.bleu_score import corpus_bleu
  score = corpus_bleu(references, candidates)
  return score

def is_exact_match(complex_sentence, simple_sentence):
    return complex_sentence == simple_sentence

def flesch_reading_ease(text):
  return textstat.flesch_reading_ease(text)

def flesch_kincaid_grade(text):
  return textstat.flesch_kincaid_grade(text)

def automated_readability_index(text):
  return textstat.automated_readability_index(text)

@lru_cache(maxsize=1)
def get_word2rank(vocab_size=50000):
    frequency_table_path = os.path.join(VARIOUS_DIR, 'enwiki_frequency_table.tsv')
    word2rank = {}
    for rank, line in enumerate(yield_lines(frequency_table_path)):
        if (rank+1) > vocab_size:
            break
        word, _ = line.split('\t')
        word2rank[word] = rank
    return word2rank

def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))


def get_avg_std(data):
  return (np.mean(data), np.std(data))


def compute_easse_metrics(data):
  compression_ratio_max = []
  compression_ratio_avg = []
  compression_ratio_easse = []
  levenshtein_distance = []
  lev_sim = []
  added_words = []
  deleted_words = []
  kept_words = []
  add_prop = []
  del_prop = []
  exact_match = []
  expert_fk_ease = []
  expert_fk_grade = []
  expert_ari = []
  layman_fk_ease = []
  layman_fk_grade = []
  layman_ari = []
  for idx, pair in enumerate(data):
    c, s = pair
    lev_sim.append(get_levenshtein_similarity(c, s))
    compression_ratio_max.append(compression_corpus(c, s, tag='max'))
    compression_ratio_avg.append(compression_corpus(c, s, tag='avg'))
    compression_ratio_easse.append(get_compression_ratio(c, s))
    levenshtein_distance.append(get_levenshtein_distance(c, s))
    added_words.append(get_n_added_words(c,s))
    deleted_words.append(get_n_deleted_words(c,s))
    kept_words.append(get_n_kept_words(c,s))
    exact_match.append(is_exact_match(c,s))
    add_prop.append(get_additions_proportion(c, s))
    del_prop.append(get_deletions_proportion(c, s))
    expert_fk_ease.append(flesch_reading_ease(c))
    expert_fk_grade.append(flesch_kincaid_grade(c))
    expert_ari.append(automated_readability_index(c))
    layman_fk_ease.append(flesch_reading_ease(s))
    layman_fk_grade.append(flesch_kincaid_grade(s))
    layman_ari.append(automated_readability_index(s))

  print("levenstein sim: ", get_avg_std(lev_sim))
  print("easse compression ratio: ", get_avg_std(compression_ratio_easse))
  print("added words: ", get_avg_std(added_words))
  print("deleted words: ", get_avg_std(deleted_words))
  print("kept words: ", get_avg_std(kept_words))
  print("add prop: ", get_avg_std(add_prop))
  print("del prop: ", get_avg_std(del_prop))
  print("exact match: ", get_avg_std(exact_match)[0])

  print("Expert  ============================================================")

  print("fk_ease: ", get_avg_std(expert_fk_ease))
  print("fk_grade: ", get_avg_std(expert_fk_grade))
  print("ari: ", get_avg_std(expert_ari))

  print("Layman  ============================================================")

  print("fk_ease: ", get_avg_std(layman_fk_ease))
  print("fk_grade: ", get_avg_std(layman_fk_grade))
  print("ari: ", get_avg_std(layman_ari))

  return lev_sim, added_words, deleted_words, kept_words, exact_match

def get_lev_sim(data):
    lev_sim = []
    for idx, pair in enumerate(data):
        print(idx, pair)
        c, s = pair
        lev_sim.append(get_levenshtein_similarity(c, s))
    return lev_sim


def compression_ratio(data):
    compression_ratio_easse = []
    for idx, pair in enumerate(data):
        c, s = pair
        compression_ratio_easse.append(get_compression_ratio(c, s))
    return compression_ratio_easse

def get_reading_scores(data):
    expert_fk_grade = []
    expert_ari = []
    layman_fk_grade = []
    layman_ari = []
    for idx, pair in enumerate(data):
        c, s = pair
        expert_fk_grade.append(flesch_kincaid_grade(c))
        expert_ari.append(automated_readability_index(c))
        layman_fk_grade.append(flesch_kincaid_grade(s))
        layman_ari.append(automated_readability_index(s))
    return expert_fk_grade, expert_ari, layman_fk_grade, layman_ari

dir = './Datasets/annotated_data/'
file = 'tim_akina_final_UMLS.csv'
df = pd.read_csv(dir+file, encoding='unicode_escape', engine='python')
#df.drop(columns=['Unnamed: 0'])
text_pair = [[x, y] for x, y in zip(list(df['Expert']), list(df['Simple']))]

#lev_sim, added_words, deleted_words, kept_words, exact_match = compute_easse_metrics(text_pair)

df['sim'] = get_lev_sim(text_pair)

sentence_sim = []

for idx, row in df.iterrows():
    sentence_sim.append(metrics.sent_align([row['Expert'], row['Simple']]))
    print(idx)
    
df['sentence_sim'] = sentence_sim
df['compression'] = compression_ratio(text_pair)

expert_fk_grade, expert_ari, layman_fk_grade, layman_ari = get_reading_scores(text_pair)

df['expert_fk_grade'] = expert_fk_grade
df['expert_ari'] = expert_ari
df['layman_fk_grade'] = layman_fk_grade
df['layman_ari'] = layman_ari 


df.to_csv(dir + file, index=False)



