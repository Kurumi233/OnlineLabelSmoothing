# OnlineLabelSmoothing

This is a re-implementation of Online Label Smoothing. The code is written based on my understanding of the paper. If there's any bug in my code, please tell me in the **Issues** page.

## Usage

```python
from OLS import OnlineLabelSmoothing

ols_loss = OnlineLabelSmoothing(num_classes=1000, use_gpu=True)

# Training
for epoch in range(total_epoch):
    # train()
    # test()
    ols_loss.update()

# Saving
torch.save({'ols': ols_loss.matrix.cpu().data}, 'ols.pth')
```

## Results

#### Environment

- Python 3.7
- PyTorch 1.6.0
- GPU: Tesla V100 32GB * 1

#### Other Setting

```python
num_classes: 1000
optimizer: SGD
init_lr: 0.1
weight_decay: 0.0001
momentum: 0.9
lr_gamma: 0.1
total_epoch: 250
batch_size: 256
num_workers: 20
random_seed: 2020
amp: True # automatic mixed-precision training, this function is offered by pytorch
```

#### Train

- use single gpu

```shell
python train.py --amp -s cos --loss ce ols --loss_w 0.5 0.5
```

- use multi gpus single node

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch\
--nproc_per_node=2 --master_addr 127.0.0.7 --master_port 23456\
train.py --multi-gpus 1 -nw 20 --amp -s multi --loss ce ols --loss_w 0.5 0.5
```

- use multi gpus multi nodes

```shell
# Limited computing resources
```

#### Accuracy on Validation Set of ImageNet2012

Although I used AMP(automatic mixed-precision) to speed up my training, it still took me nearly five days, so I didn't do any other experiments with ols. But there are other records of training ImageNet in my [blog](https://blog.csdn.net/u013347145/article/details/113175942).

| Model    | Loss | epoches | lr_schedule | Acc@1 | Acc@5 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| ResNet50 | CE | 250 | Multi Step [75,150,225] | 76.32 | 93.06 |
| ResNet50 | CE | 250 | COS with 5 epochs warmup | 76.95 | 93.27 |
| ResNet50 | 0.5\*CE+0.5\*OLS | 250 | Multi Step [75,150,225] | 77.27 | 93.47 |
| ResNet50 | 0.5\*CE+0.5\*OLS | 250 | COS with 5 epochs warmup | 77.79 | 93.79 |


#### Reference

- [Delving Deep into Label Smoothing](https://arxiv.org/pdf/2011.12562.pdf)
- https://github.com/zhangchbin/OnlineLabelSmooth

