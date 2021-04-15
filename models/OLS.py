"""
Reference:
    paper: Delving Deep into Label Smoothing.
"""
import torch
import torch.nn as nn
from collections import Counter


class OnlineLabelSmoothing(nn.Module):
    def __init__(self, num_classes=10, use_gpu=False):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes))
        self.grad = torch.zeros((num_classes, num_classes))
        self.count = torch.zeros((num_classes, 1))
        if use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

    def forward(self, x, target):
        target = target.view(-1,)
        logprobs = torch.log_softmax(x, dim=-1)

        softlabel = self.matrix[target]
        loss = (- softlabel * logprobs).sum(dim=-1)

        if self.training:
            # accumulate correct predictions
            p = torch.softmax(x.detach(), dim=1)
            _, pred = torch.max(p, 1)
            correct_index = pred.eq(target)
            correct_p = p[correct_index]
            correct_label = target[correct_index].tolist()

            self.grad[correct_label] += correct_p
            for k, v in Counter(correct_label).items():
                self.count[k] += v

        return loss.mean()

    def update(self):
        index = torch.where(self.count > 0)[0]
        self.grad[index] = self.grad[index] / self.count[index]
        # reset matrix and update
        nn.init.constant_(self.matrix, 0.)
        norm = self.grad.sum(dim=1).view(-1, 1)
        index = torch.where(norm > 0)[0]
        self.matrix[index] = self.grad[index] / norm[index]
        # reset
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)


if __name__ == '__main__':
    import random
    ols = OnlineLabelSmoothing(num_classes=6, use_gpu=False)
    x = torch.randn((10, 6))
    y = torch.LongTensor(random.choices(range(6), k=10))

    l = ols(x, y)
    print('ols:', l)
    ols.update()
    """ Compare the time performance, 100-batchs
    y = torch.LongTensor(random.choices(range(100), k=256)).cuda()
    count1 = torch.zeros((100, 1)).cuda()
    count2 = torch.zeros((100, 1)).cuda()
    t1 = time.time()
    for i in range(100):
        for k, v in Counter(y.tolist()).items():
            count1[k] += v
    t2 = time.time()

    print(t2 - t1)
    # 0.19294953346252441

    t3 = time.time()
    for i in range(100):
        for k in y:
            count2[k] += 1
    t4 = time.time()
    print(t4 - t3)
    # 2.9499971866607666

    print((count1-count2).sum())
    """