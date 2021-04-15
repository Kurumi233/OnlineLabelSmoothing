import matplotlib.pyplot as plt
import numpy as np
import re


def str2dict(line):
    d = {}
    sp = line.strip().split()
    for s in sp:
        k, v = s.split(':')
        d[k] = float(v)
    return d


def dict_append(root, x):
    for k, v in x.items():
        if k not in root:
            root[k] = []
        root[k].append(v)
    return root


def plot(res, savepath='.', name='loss', mode='train', best=None):
    if(isinstance(res, dict)):
        for k, v in res.items():
            plt.plot(range(len(v)), v, label=k)
        plt.legend()
    if(isinstance(res, list)):
        pass
    plt.savefig('{}/{}_{}'.format(savepath, mode, name), bbox_inches='tight')
    plt.close()


def plot_result(txt, savepath='.'):
    epoch_pattern = r'epoch:(.*?) '
    lr_pattern = r'lr:(.*?) '
    loss_pattern = r'loss\[(.*?)\]'
    acc_pattern = r'acc\[(.*?)\]'
    epoch = []
    lr = []
    train_loss = {}
    train_acc = {}
    val_loss = {}
    val_acc = {}
    with open(txt, 'r')as f:
        for i in f.readlines():
            if i[0] == '#':
                print(i)
            else:
                epoch.append(int(re.search(epoch_pattern, i).group(1)))
                lr.append(float(re.search(lr_pattern, i).group(1)))
                loss = re.findall(loss_pattern, i)
                acc = re.findall(acc_pattern, i)
                train_loss = dict_append(train_loss, str2dict(loss[0]))
                train_acc = dict_append(train_acc, str2dict(acc[0]))
                val_loss = dict_append(val_loss, str2dict(loss[1]))
                val_acc = dict_append(val_acc, str2dict(acc[1]))
        # print(train_loss)
        # print(val_loss)
        plot(train_loss, savepath=savepath)
        plot(val_loss, savepath=savepath, mode='test')


if __name__ == '__main__':
    txt = './cifar100/resnet18_avg_linear_0_ce_ols/log.txt'
    plot_result(txt)