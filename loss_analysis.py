import pickle
import matplotlib.pyplot as plt


def get_training_loss(dir):
    losses = []
    for i in range(30):
        file = dir + 'result_for_epoch_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            e = pickle.load(f)
            total_loss = 0
            for k, v in e.items():
                if k != 'eval':
                    total_loss += v['loss_per_batch']
            total_loss /= (len(e.keys()) - 1)
            losses.append(total_loss)
    return losses

def get_eval_loss(dir):
    losses = []
    for i in range(30):
        file = dir + 'result_for_epoch_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            e = pickle.load(f)
            losses.append(e['eval']['eval_loss'])
    return losses


dir = './results/t5_large/merged_outputs/exc_EaSa_alt_input_format_single_angle/'
losses_large_train = get_training_loss(dir)
losses_large_eval = get_eval_loss(dir)
plt.plot(losses_large_train, label = 'train T5-large')
plt.plot(losses_large_eval, label = 'eval T5-large')
plt.legend()
plt.savefig('./results/t5_large/merged_outputs/t5_single_angle_no_simple.png', dpi=150)


