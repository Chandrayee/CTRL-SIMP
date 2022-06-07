import pickle
from collections import Counter

with open('./results/t5_small/merged_outputs/exc_EaSa_alt_input_format/result_for_epoch_0.pkl', 'rb') as f:
    data = pickle.load(f)
    
angles = []

for k, x in data.items():
    if k != 'eval':
        angles.append([y['angle'] for y in x['res_per_example']])
        
angle_set = Counter(angles)
print(angle_set)