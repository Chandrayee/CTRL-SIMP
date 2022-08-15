import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chkpt', type = int, default = 0)
args = parser.parse_args()

dir = './results/t5_large/merged_outputs/exc_EaSa_alt_input_format/multiangle/unannotated/test_sampling/'

#for j in range(0,30):
    #print(j)
dflist = []
for i in range(0, 725, 4):
    locals()['df_'+str(i)] = pd.read_csv(dir + 'part_files/eval_exc_EaSa_alt_input_format_'+str(args.chkpt)+'_batchno'+str(i)+'.csv')
    dflist.append(locals()['df_'+str(i)])

df = pd.concat(dflist)

print(len(df))
print(df.head(10))

df.to_csv(dir + 'eval_exc_EaSa_alt_input_format_epoch'+str(args.chkpt)+'.csv', index=False)
