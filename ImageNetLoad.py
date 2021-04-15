from torch.utils.data import dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


img_root = '/path/to/ImageNet/Image'
devkit   = '/path/to/ImageNet/devkit/caffe_ilsvrc12'


# trans = {
#     'train':
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ]),
#     'val':
#     transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])}

trans = {
    'train':
        A.Compose([
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            # A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0, always_apply=False, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
    'val':
        A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])}


class ImageNet(dataset.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'val']
        txt = os.path.join(devkit, '%s.txt' % mode)
        self.dataroot = os.path.join(img_root, mode, 'images')

        fpath = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, label = i.strip().split(' ')
                fpath.append(os.path.join(self.dataroot, fp))
                labels.append(int(label))

        self.fpath = fpath
        self.labels = labels
        self.mode = mode
        self.trans = trans[mode]

    def __getitem__(self, index):
        fp = self.fpath[index]
        label = self.labels[index]

        img = Image.open(fp).convert('RGB')
        
        img = np.array(img)
        if self.trans is not None:
            img = self.trans(image=img)["image"]

        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import warnings
    import piexif

    warnings.filterwarnings('error')

    dataset = ImageNet(mode='train')
    print(len(dataset))

    loader = DataLoader(dataset=dataset,
                           batch_size=256,
                           shuffle=False,
                           num_workers=10,
                           pin_memory=True)

    for idx, (data, label) in enumerate(loader):
        print(idx)