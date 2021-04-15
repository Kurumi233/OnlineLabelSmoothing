import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25, reduce=True, logits=True):
        """
        :param gamma:
        :param alpha: class weights
        :param reduce:
        :param logits: if Sigmoid applied
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce
        self.logits = logits

    def forward(self, x, target):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(x, target, reduce=False)
        else:
            bce = F.binary_cross_entropy(x, target, reduce=False)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Code: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/focal_loss.py
    """
    def __init__(self, gamma=2., eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        logp = self.ce(x, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        return loss.mean()


class FocalLossv2(nn.Module):
    '''Multi-class Focal loss implementation'''
    """
    Code: https://github.com/ashawkey/FocalLoss.pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, x, target):
        logpt = F.log_softmax(x, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLossv3(nn.Module):
    """
    Code: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=2., alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    x = torch.randn((4, 10))
    y = torch.LongTensor([1, 2, 3, 0])

    f1 = FocalLoss()(x, y)
    print('Focal1:', f1)
    f2 = FocalLossv2()(x, y)
    print('Focal2:', f2)
    f3 = FocalLossv3()(x, y)
    print('Focal2:', f3)

