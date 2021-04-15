from __future__ import division, absolute_import
import torch
import matplotlib.pyplot as plt


def plot_result():
    pass


def accuracy(out, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)

        _, pred = out.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k / bs)
        return res


class AverageMeter(object):
    def __init__(self, name, fmt):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, bs):
        self.val = val
        self.sum += val * bs
        self.count += bs
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg})' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MultiLossAverageMeter(object):
    def __init__(self, loss):
        """
        :param loss: list, names of losses
        """
        self.dict = {}
        self.loss = loss
        self.reset()

    def reset(self):
        for l in self.loss:
            self.dict[l] = 0.
        self.all = 0.
        self.count = 0.

    def update(self, val, bs=1):
        """
        :param val: dict, values of each loss
        :param bs: batch size
        """
        count = self.count
        self.count += bs
        for k, v in self.dict.items():
            self.dict[k] = (v * count + val[k]) / self.count

    def __repr__(self, avg=True):
        fmtstr = ''
        for k, v in self.dict.items():
            if avg:
                fmtstr += '{}:{:.4f} '.format(k, v)
            else:
                fmtstr += '{}:{:.4f} '.format(k, v * self.count)

        return fmtstr.rstrip()