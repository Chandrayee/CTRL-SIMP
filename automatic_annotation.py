import strmatching
import dataprep
import argparse
import difflib
import pandas as pd

def get_replacement(original_t, rewrit_t):
    a, b = original_t.split(), rewrit_t.split()
    s, r = original_t.lower().split(), rewrit_t.lower().split()
    diff_str = difflib.SequenceMatcher(None, s, r)
    diff_str.a = a
    diff_str.b = b
    diff_anno = strmatching.show_diff(diff_str)
    return diff_anno

def run_rep(expert_text, layman_text, sim):
    annotated = []
    for i, (e, l) in enumerate(zip(expert_text, layman_text)):
        diff_e_l = get_replacement(e, l)
        annotated.append(diff_e_l)
        #strmatching.get_codes(e.lower().split(), l.lower().split())
        print("===========================================\n")
        print(sim[i], " : ", diff_e_l)
    return annotated
    
def test_matcher():
    '''
    tags: <rep>, <del>, <ins>
    '''
    #s= "Since then , the term `` regression '' has taken on a variety of meanings , and it may be used by modern statisticians to describe phenomena of sampling bias which have little to do with Galton 's original observations in the field of genetics ."
    #r = "Since then , the term `` regression '' has taken on different meanings , and it may be used by modern statisticians to describe phenomena of sampling bias which have little to do with Galton 's original observations in the field of genetics ."
    s = "Further inquiry should determine whether the pain is superficial or deep - whether it occurs primarily at the vaginal outlet or vaginal barrel or upon deep thrusting against the cervix ."
    r = "Inquiry should determine whether the pain is superficial or deep - whether it occurs primarily at the vaginal outlet or vaginal barrel or upon deep thrusting against the cervix ."
    #strmatching.get_codes(s, r)
    print("===========================================\n")
    print(get_replacement(s, r))


if __name__ == "__main__":
    #test_matcher()
    parser = argparse.ArgumentParser(description='process inputs')
    parser.add_argument('--dataset', type=str, default='simpwiki')
    parser.add_argument('--datatype', type=str, default='expert_fully_aligned')
    args = parser.parse_args()

    
    if args.dataset == 'simpwiki':
        dir = "./Datasets/" + args.dataset 
        if args.datatype == 'sample':
            file = 'mixed_samples.csv'
        datapath = dir + '/' + file
        data = pd.read_csv(datapath)
        expert_text = data.expert
        layman_text = data.layman
        sim = data.sim
        annotated = run_rep(expert_text, layman_text, sim)

    df_annotated = pd.DataFrame(annotated, columns = ['annotated'])
    df = pd.concat([data, df_annotated], axis = 1, join = 'inner')
    df.to_csv(dir + '/' + 'annotated_mixed.csv')
        
    #expert_text, layman_text = dataprep.parse_data(args.dataset, args.datatype)
    #run_rep(expert_text, layman_text)'''
