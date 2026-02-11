"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


from self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from self_training.base import ImageClassifier, DynamicThresholdingModule
from vision.transforms import MultipleApply
from utils.metric import accuracy
from utils.meter import AverageMeter, ProgressMeter
from utils.data import ForeverDataIterator
from utils.logger import CompleteLogger
from self_training.supconloss import SupConLoss
from self_training.kl_div import KLDivergence
import utils_ as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# python train.py -d wm -a resnet34 --epochs 20 -i 500

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    weak_augment = utils.get_train_transform2(args.train_resizing, random_horizontal_flip=True)
    strong_augment = utils.get_train_transform2(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment)
    labeled_train_transform = MultipleApply([weak_augment, strong_augment])
    unlabeled_train_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform2(args.val_resizing)
    print('labeled_train_transform: ', labeled_train_transform)
    print('unlabeled_train_transform: ', unlabeled_train_transform)
    print('val_transform:', val_transform)
    source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test, source_save, target_save = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=unlabeled_train_transform,
                          seed=args.seed)
    print("source_dataset_size: ", len(source_dataset))
    print("target_dataset_size: ", len(target_dataset))
    print('target_dataset_unl_size: ', len(target_dataset_unl))
    print("target_dataset_val_size: ", len(target_dataset_val))

    source_train_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    labeled_train_loader = DataLoader(target_dataset, batch_size=min(args.batch_size,len(target_dataset)), shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    unlabeled_train_loader = DataLoader(target_dataset_unl, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    source_train_iter = ForeverDataIterator(source_train_loader)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)

    val_loader = DataLoader(target_dataset_val, batch_size=min(args.batch_size,len(target_dataset_val)), shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(target_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    source_save_loader = DataLoader(source_save, batch_size=min(args.batch_size,len(target_dataset_val)), shuffle=False, num_workers=args.workers)
    target_save_loader = DataLoader(target_save, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    # print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone,pretrained=False)
    num_classes = target_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, width=args.width,
                                 pool_layer=pool_layer, finetune=args.finetune).to(device)
    # print(classifier)

    thresholding_module = DynamicThresholdingModule(0.7, num_classes,
                                                    len(target_dataset_unl), device=device)

    # define optimizer and lr scheduler
    if args.lr_scheduler == 'exp':
        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                        nesterov=True)
        lr_scheduler = utils.get_cosine_scheduler_with_warmup(optimizer, args.epochs * args.iters_per_epoch)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    best_avg = 0.0
    best_acc_test = 0.0
    best_dict = None
    pseudo_d = {'n_total':[],'n_correct':[],'n_pseudo_labels':[], 'acc':[],'threshold':[]}
    feat_save = None
    feature_distance = {'e_distence_back':[],'e_distence_bottl':[],'acc':[]}

    for epoch in range(args.epochs):
        # print(lr_scheduler.get_last_lr())

        # train for one epoch
        dynamic_threshold, loss = train(thresholding_module, source_train_iter,labeled_train_iter, unlabeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args)
        print(dynamic_threshold)

        # evaluate on validation set
        acc1, avg, D, _= utils.validate(val_loader, classifier, args, device, num_classes)
        acct, avgt, Dt, pseudo= utils.validate(test_loader, classifier, args, device, num_classes, threshold = dynamic_threshold.cpu() #args.threshold 
                    )

        # ft, e_distence = utils.save_feat(source_save_loader, target_save_loader, classifier, args, device, num_classes)
        # e_distence['acc'] = acct

        # remember best acc@1 and save checkpoint
        # torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 >= best_acc1 or abs(acc1-best_acc1)<0.001:
        # if acct >= best_acc_test:
            best_acc_test = acct
            best_dict = Dt
            # feat_save = ft
        #     shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        best_avg = max(avg, best_avg)
        # for p in pseudo_d.keys():
        #     pseudo_d[p].append(pseudo[p])
        # for p in feature_distance.keys():
        #     feature_distance[p].append(e_distence[p])

        # np.save('/home/user/disk/yiru/feature_visual/dst/dst-static2.npy',static)
    # np.save('/home/user/disk/yiru/feature_visual/dst/losses-cca.npy',np.array(losses))
    # np.save('/home/user/disk/yiru/feature_visual/dst/dst-nosample-distance-'+str(round(best_acc_test,2))+'.npy',feature_distance)
    print("best_acc_val = {:3.1f}".format(best_acc1))
 
    print('best acc test %f' % (best_acc_test))
    print('best acc per: ', *best_dict['acc_per'])
    print('best f1: ', best_dict['f1'])
    print('best f1 per: ', *best_dict['f1_per'])
    print('best recall: ', best_dict['recall'])
    print('best precision: ', best_dict['precision'])
    

    logger.close()


def train(thresholding_module, source_train_iter: ForeverDataIterator,labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':3.2f')
    supconlosses = AverageMeter('supconloss Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, supconlosses, cls_accs, pseudo_label_accs,
         pseudo_label_ratios],
        prefix="Epoch: [{}]".format(epoch))

    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
    kl_criterion = KLDivergence(tau=2)
    worst_case_estimation_criterion = WorstCaseEstimationLoss(args.eta_prime).to(device)
    supcon_criterion = SupConLoss(temperature=0.07).to(device)
    # switch to train mode
    model.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_s, x_s_strong), labels_s = next(source_train_iter)
        x_s = x_s.to(device)
        x_s_strong = x_s_strong.to(device)
        labels_s = labels_s.to(device)

        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device)
        x_l_strong = x_l_strong.to(device)
        labels_l = labels_l.to(device)

        (x_u, x_u_strong), labels_u = next(unlabeled_train_iter)
        x_u = x_u.to(device)
        x_u_strong = x_u_strong.to(device)
        labels_u = labels_u.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()
        start = time.time()
        # compute output
        x = torch.cat((x_l, x_u), dim=0)
        
        outputs, midfeat, _, _ = model(x)
        y_l, y_u = outputs[:len(x_l)],outputs[len(x_l):]
        # y_l_feat, y_u_feat = midfeat[:len(x_l)],midfeat[len(x_l):]
        _, y_s_feat, _, y_s = model(x_s)     #源域

        # ==============================================================================================================
        # cross entropy loss (weak augment)
        # ==============================================================================================================
        cls_loss_weak = F.cross_entropy(y_l, labels_l) + F.cross_entropy(y_s, labels_s)
        cls_loss_weak.backward()


        # ==============================================================================================================
        # cross entropy loss (strong augment)
        # ==============================================================================================================
        y_l_strong, _, _, _ = model(x_l_strong)
        _, feat_s, _, y_s_strong = model(x_s_strong)
        cls_loss_strong = args.trade_off_cls_strong * (F.cross_entropy(y_l_strong, labels_l) + F.cross_entropy(y_s_strong, labels_s))
        # cls_loss_strong.backward()

        # ==============================================================================================================
        # self training loss
        # ==============================================================================================================
        _, feat_u, y_u_strong, _ = model(x_u_strong)
        # self_training_loss, mask, pseudo_labels = self_training_criterion(y_u_strong, y_u)
        # self_training_loss = args.trade_off_self_training * self_training_loss

        #self_training
        confidence, pseudo_labels = F.softmax(y_u.detach(), dim=1).max(dim=1)
        warmup = (epoch<=5)   #-1,0,1,3,7,11,15,19
        status = thresholding_module.get_threshold(confidence, pseudo_labels, warmup)
        dynamic_threshold = status['threshold']
        mask = (confidence > dynamic_threshold[pseudo_labels]).float()
        # mask used for updating learning status
        self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # self_training_loss.backward()

        #-----------sca
        _, feat_s_c, _, y_s2 = model(x_s_strong, True)   #源域cross
        _, feat_u_c, y_u2, _ = model(x_u_strong, True)    #目标域cross
        # x_ls_strong = torch.cat((x_l_strong, x_s_strong), dim=0)
        sampleloss_hard = (F.cross_entropy(y_u2, pseudo_labels, reduction='none') * mask).mean() + F.cross_entropy(y_s2, labels_s)
        sampleloss_soft = kl_criterion(y_s2, y_s_strong) + kl_criterion(y_u2, y_u_strong)
        supconloss = (sampleloss_hard + sampleloss_soft) *0.5

        #cca
        feat_s_c, _ = feat_s_c
        feat_s, _ = feat_s
        _, feat_u_c = feat_u_c
        _, feat_u = feat_u
        suorce, suorce_y = torch.cat((feat_s_c,feat_s),0), torch.cat((labels_s,labels_s),0)
        target, target_y = torch.cat((feat_u_c,feat_u),0), torch.cat((pseudo_labels,pseudo_labels),0)
        loss_s = supcon_criterion(suorce.view([suorce.shape[0],1,-1]), suorce_y).mean() 
        loss_t = supcon_criterion(target.view([target.shape[0],1,-1]), target_y).mean() 
        supconloss = (loss_t + loss_s) * 0.5 + supconloss  

        if warmup:
            (cls_loss_strong + self_training_loss).backward()
        else:
            (cls_loss_strong +supconloss+ self_training_loss).backward()
        

        # print(time.time()-start)s
        # measure accuracy and record loss
        cls_loss = cls_loss_strong + cls_loss_weak
        cls_losses.update(cls_loss.item(), batch_size)
        loss = cls_loss + self_training_loss + supconloss 
        
        losses.update(loss.item())
        #wce_losses.update(wce_loss.item(), batch_size)
        supconlosses.update(supconloss.item())
        self_training_losses.update(self_training_loss.item())

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # ratio of pseudo labels
        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels / batch_size
        pseudo_label_ratios.update(ratio.item() * 100, batch_size)
        # pseudo_d['pseudo_ratio'].append(ratio.item() * 100)

        # accuracy of pseudo labels
        if n_pseudo_labels > 0:
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            n_correct = (pseudo_labels == labels_u).float().sum()
            pseudo_label_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)

        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()
        model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return dynamic_threshold, loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debiased Self-Training for Semi Supervised Learning')
    # dataset parameters
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=3, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--train-resizing', default='default', type=str)
    parser.add_argument('--val-resizing', default='default', type=str)
    parser.add_argument('--norm-mean', default=(0.485, 0.456, 0.406), type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=(0.229, 0.224, 0.225), type=float, nargs='+',
                        help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet34)')
    parser.add_argument('--width', default=2048, type=int,
                        help='width of the pseudo head and the worst-case estimation head')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')
    # training parameters
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
    parser.add_argument('--trade-off-self-training', default=1, type=float,
                        help='the trade-off hyper-parameter of self training loss')
    parser.add_argument('--eta', default=1, type=float,
                        help='the trade-off hyper-parameter of adversarial loss')
    parser.add_argument('--eta-prime', default=2, type=float,
                        help="the trade-off hyper-parameter between adversarial loss on labeled data "
                             "and that on unlabeled data")
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='confidence threshold')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='exp', type=str, choices=['exp', 'cos'],
                        help='learning rate decay strategy')
    parser.add_argument('--lr-gamma', default=0.0002, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='dst', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    args = parser.parse_args()
    main(args)
