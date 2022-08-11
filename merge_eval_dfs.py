import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 4)
parser.add_argument('--chkpt', type = int, default = 1)
args = parser.parse_args()

dir = './results/t5_large/merged_outputs/exc_EaSa_alt_input_format_single_angle/ea2sa/longer_training/bs'+str(args.batch_size)+'/test/'

#for chkpt in range(0,60):
dflist = []
for i in range(0, 300, 4):
    locals()['df_'+str(i)] = pd.read_csv(dir + 'part_files/eval_exc_EaSa_alt_input_format_single_angle_'+str(args.chkpt)+'_batchno'+str(i)+'.csv')
    dflist.append(locals()['df_'+str(i)])

df = pd.concat(dflist)

print(len(df))
print(df.head(10))

df.to_csv(dir + 'eval_exc_EaSa_alt_input_format_single_angle_epoch'+str(args.chkpt)+'.csv', index=False)
