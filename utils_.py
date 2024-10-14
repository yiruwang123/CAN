"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import math
import sys
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataset import Subset, ConcatDataset
import torchvision.transforms as T
import timm
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
from classifier import Classifier
import vision.datasets as datasets
import vision.models as models
from utils.metric import accuracy, ConfusionMatrix
from utils.meter import AverageMeter, ProgressMeter
from sklearn.model_selection import train_test_split

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained=False, pretrained_checkpoint=None):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrained)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, num_samples_per_class, root, labeled_train_transform, val_transform,
                unlabeled_train_transform=None, seed=0):
    if unlabeled_train_transform is None:
        unlabeled_train_transform = labeled_train_transform

    
    if dataset_name == "wm":   #SSDA
        
        df=np.load('wm811k-64.npz')
        # df=np.load('/home/user/disk/yiru/mixed2m38-64-single.npz')
        source = df['data']
        source_label = df['label']
        class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
        num_classes = len(class_names)

        # df=np.load('/home/user/disk/yiru/wm811k-64.npz')
        df=np.load('mixed2m38-64-single.npz')
        target = df['data']
        target_label = df['label']

        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, num_classes,
                                                   target_label, seed=0)
        target_l = target[labeled_idxes]
        target_label_l = target_label[labeled_idxes]
        target = target[unlabeled_idxes]
        target_label = target_label[unlabeled_idxes]

        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, num_classes,
                                                        target_label, seed=0)
        target_val = target[labeled_idxes]
        target_label_val = target_label[labeled_idxes]
        target = target[unlabeled_idxes]
        target_label = target_label[unlabeled_idxes]

        source_dataset = ImageDataset2(64, source, classes=source_label,transform=labeled_train_transform,num_classes=num_classes)
        target_dataset = ImageDataset2(64, target_l, classes=target_label_l,transform=labeled_train_transform,num_classes=num_classes)
        target_dataset_val = ImageDataset2(64, target_val, classes=target_label_val,transform=val_transform,num_classes=num_classes)
        target_dataset_unl = ImageDataset2(64, target, classes=target_label,transform=unlabeled_train_transform,num_classes=num_classes)
        target_dataset_test = ImageDataset2(64, target, classes=target_label,transform=get_test_transform2(),num_classes=num_classes)

        #### save_feat
        labeled_idxes2, unlabeled_idxes2 = x_u_split(400, num_classes,
                                                   source_label, seed=0)
        source_feat = source[labeled_idxes2]
        source_feat_l = source_label[labeled_idxes2]

        labeled_idxes2, unlabeled_idxes2 = x_u_split(400, num_classes,
                                                   target_label, seed=0)
        target_feat = target[labeled_idxes2]
        target_feat_l = target_label[labeled_idxes2]

        source_save = ImageDataset2(64, source_feat, classes=source_feat_l,transform=val_transform,num_classes=num_classes)
        target_save = ImageDataset2(64, target_feat, classes=target_feat_l,transform=val_transform,num_classes=num_classes)
        ####

        return source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test, source_save, target_save

    else:
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        # create labeled and unlabeled splits
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, base_dataset.num_classes,
                                                   base_dataset.targets, seed=seed)
        # labeled subset
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        labeled_train_dataset.num_classes = base_dataset.num_classes
        # unlabeled subset
        base_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=True)
        unlabeled_train_dataset = Subset(base_dataset, unlabeled_idxes)
        val_dataset = dataset(root=root, split='test', download=True, transform=val_transform)
    return labeled_train_dataset, unlabeled_train_dataset, val_dataset


def x_u_split(num_samples_per_class, num_classes, labels, seed):
    """
    Construct labeled and unlabeled subsets, where the labeled subset is class balanced. Note that the resulting
    subsets are **deterministic** with the same random seed.
    """
    labels = np.array(labels)
    assert num_samples_per_class * num_classes <= len(labels)
    random_state = np.random.RandomState(seed)

    # labeled subset
    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        if len(ith_class_idxes) < num_samples_per_class:
            ith_class_idxes = random_state.choice(ith_class_idxes, len(ith_class_idxes), False)
        else:
            ith_class_idxes = random_state.choice(ith_class_idxes, num_samples_per_class, False)
        labeled_idxes.extend(ith_class_idxes)

    # unlabeled subset
    unlabeled_idxes = [i for i in range(len(labels)) if i not in labeled_idxes]
    return labeled_idxes, unlabeled_idxes



def get_train_transform2(resizing='default', random_horizontal_flip=True, auto_augment=None
                        ):
    transforms = []
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(64 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in [0.5]]),
            interpolation=Image.BILINEAR
        )
        auto_augment="rand-m5-n2-mstd1"
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    transforms.extend([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    return T.Compose(transforms)

def get_val_transform2(resizing='default'):

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

def get_test_transform2(resizing='default'):

    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

def convert_dataset(dataset):
    """
    Converts a dataset which returns (img, label) pairs into one that returns (index, img, label) triplets.
    """

    class DatasetWrapper:

        def __init__(self):
            self.dataset = dataset

        def __getitem__(self, index):
            return index, self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper()


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions


def get_cosine_scheduler_with_warmup(optimizer, T_max, num_cycles=7. / 16., num_warmup_steps=0,
                                     last_epoch=-1):
    """
    Cosine learning rate scheduler from `FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence (NIPS 2020) <https://arxiv.org/abs/2001.07685>`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        num_cycles (float): A scalar that controls the shape of cosine function. Default: 7/16.
        num_warmup_steps (int): Number of iterations to warm up. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, T_max - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
def validate(val_loader, model, args, device, num_classes, threshold=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(val_loader),
    #     [batch_time, losses, top1, top5],
    #     prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    y_true = torch.Tensor([])
    y_pred  = torch.Tensor([])

    n_pseudo_labels = 0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            confidence, predicted = F.softmax(output.cpu(), dim=1).max(dim=1)
            
            y_true = torch.cat([y_true,target.cpu()],dim = 0)
            y_pred = torch.cat([y_pred,predicted],dim = 0)

            #####pseudo_label统计
            if threshold != None:
                th = threshold[predicted]
                mask = (confidence > th).float()
                n_pseudo_labels += mask.sum()
                n_total += len(images)
                pseudo_labels = predicted * mask - (1 - mask)
                n_correct += (pseudo_labels == target.cpu()).float().sum()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
        
        f1 = metrics.f1_score(y_true, y_pred,average='macro')*100
        f1_per = metrics.f1_score(y_true, y_pred,average=None)*100
        recall = metrics.recall_score(y_true, y_pred,average='macro')*100
        precision = metrics.precision_score(y_true, y_pred,average='macro')*100
        C2= confusion_matrix(y_true, y_pred)
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        acc_global, acc_per_class, iu = confmat.compute()
        mean_cls_acc = acc_per_class.mean().item() * 100
        # print(' * Mean Cls {:.3f}'.format(mean_cls_acc))

        pseudo = {'n_total':n_total,'n_correct':n_correct,'n_pseudo_labels':n_pseudo_labels, 'acc':top1.avg,'threshold':threshold}
        D = {'f1':f1,'f1_per':f1_per,'acc_per':np.array(acc_per_class.cpu())*100,'recall':recall,'precision':precision,'C2':C2}

    return top1.avg, mean_cls_acc, D, pseudo

def save_feat(source_loader, target_loader, model, args, device, num_classes):
    model.eval()
    with torch.no_grad():
        # end = time.time()
        feat_s_backbone = torch.Tensor([])
        feat_s_bottl = torch.Tensor([])
        label  = torch.Tensor([])
        for i, (images, target) in enumerate(source_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            f, f_s, _ = model(images, save_feat=True)

            feat_s_backbone = torch.cat([feat_s_backbone,f.cpu()],dim = 0)
            feat_s_bottl = torch.cat([feat_s_bottl,f_s.cpu()],dim = 0)
            label = torch.cat([label,target.cpu()],dim = 0)
        dict_ = {'feat_s_backbone':feat_s_backbone.numpy(),'feat_s_bottl':feat_s_bottl.numpy(),'label_s':label.numpy()}
        # np.save('/home/user/disk/yiru/feature_visual/dst/source-9.1.npy',dict_)
        mean_s_backbone = torch.mean(feat_s_backbone,0)
        mean_s_bottl = torch.mean(feat_s_bottl,0)

        feat_s_backbone = torch.Tensor([])
        feat_s_bottl = torch.Tensor([])
        label  = torch.Tensor([])
        for i, (images, target) in enumerate(target_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            f, _, f_t = model(images, save_feat=True)

            feat_s_backbone = torch.cat([feat_s_backbone,f.cpu()],dim = 0)
            feat_s_bottl = torch.cat([feat_s_bottl,f_t.cpu()],dim = 0)
            label = torch.cat([label,target.cpu()],dim = 0)
        # dict2 = {'feat_t_backbone':feat_s_backbone.numpy(),'feat_t_bottl':feat_s_bottl.numpy(),'label_t':label.numpy()}
        dict2 = {'feat_t_backbone':feat_s_backbone.numpy(),'feat_t_bottl':feat_s_bottl.numpy(),'label_t':label.numpy()}
        # np.save('/home/user/disk/yiru/feature_visual/dst/target-1.npy',dict_)
        dict_.update(dict2)
        mean_t_backbone = torch.mean(feat_s_backbone,0)
        mean_t_bottl = torch.mean(feat_s_bottl,0)

        pdist = nn.PairwiseDistance(p=2)
        e_distence_back = pdist(mean_s_backbone, mean_t_backbone) 
        e_distence_bottl = pdist(mean_s_bottl, mean_t_bottl)         
    return dict_, {'e_distence_back':e_distence_back,'e_distence_bottl':e_distence_bottl}

def save_feat2(source_loader, target_loader, model, args, device):
    model.eval()
    with torch.no_grad():
        # end = time.time()
        feat_s_backbone = torch.Tensor([])
        label  = torch.Tensor([])
        for i, (images, target) in enumerate(source_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            y, f = model(images)

            feat_s_backbone = torch.cat([feat_s_backbone,f.cpu()],dim = 0)
            label = torch.cat([label,target.cpu()],dim = 0)
        dict_ = {'feat_s_backbone':feat_s_backbone.numpy(),'label_s':label.numpy()}
        mean_s_backbone = torch.mean(feat_s_backbone,0)

        feat_s_backbone = torch.Tensor([])
        label  = torch.Tensor([])
        for i, (images, target) in enumerate(target_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            y, f = model(images)

            feat_s_backbone = torch.cat([feat_s_backbone,f.cpu()],dim = 0)
            label = torch.cat([label,target.cpu()],dim = 0)
        dict2 = {'feat_t_backbone':feat_s_backbone.numpy(),'label_t':label.numpy()}
        dict_.update(dict2)
        mean_t_backbone = torch.mean(feat_s_backbone,0)

        pdist = nn.PairwiseDistance(p=2)
        e_distence_back = pdist(mean_s_backbone, mean_t_backbone)         
    return dict_, {'e_distence_back':e_distence_back}


def empirical_risk_minimization(labeled_train_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device)
        x_l_strong = x_l_strong.to(device)
        labels_l = labels_l.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        # cross entropy loss on both weak augmented and strong augmented samples
        loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)
        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

from torch.utils.data import DataLoader, Dataset
import random
class ImageDataset2(Dataset):
    def __init__(
        self,
        resolution,
        data,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        transform=None,target_transform=None,
        num_classes = None
    ):
        super().__init__()
        self.resolution = resolution
        self.data = data
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.transform = transform
        self.target_transform=target_transform
        self.random_flip = random_flip
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        target = self.local_classes[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
