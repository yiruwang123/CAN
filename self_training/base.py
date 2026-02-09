"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.grl import WarmStartGradientReverseLayer
from modules.classifier import Classifier
import torch

class ImageClassifier(Classifier):
    r"""
    Classifier with non-linear pseudo head :math:`h_{\text{pseudo}}` and worst-case estimation head
    :math:`h_{\text{worst}}` from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_.
    Both heads are directly connected to the feature extractor :math:`\psi`. We implement end-to-end adversarial
    training procedure between :math:`\psi` and :math:`h_{\text{worst}}` by introducing a gradient reverse layer.
    Note that both heads can be safely discarded during inference, and thus will introduce no inference cost.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer.
        width (int, optional): Hidden dimension of the non-linear pseudo head and worst-case estimation head.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - outputs: predictions of the main head :math:`h`
        - outputs_adv: predictions of the worst-case estimation head :math:`h_{\text{worst}}`
        - outputs_pseudo: predictions of the pseudo head :math:`h_{\text{pseudo}}`

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - outputs, outputs_adv, outputs_pseudo: (minibatch, `num_classes`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, width=2048, **kwargs):
        bottleneck_dim = 512
        width = 512
        
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        p = self.finetune
        self.bottleneck2 = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.bottleneck2[0].weight.data.normal_(0, 0.005)
        self.bottleneck2[0].bias.data.fill_(0.1)

        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        self.adv_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )
        self.source_head = nn.Sequential(
                    nn.Linear(self.features_dim, width),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(width, self.num_classes)
                )
        self.sclhead = nn.Sequential(
                    nn.Linear(bottleneck_dim, 128),
                )

    def forward(self, x: torch.Tensor, cross=False, save_feat=False):
        f = self.pool_layer(self.backbone(x))
        if cross:
            f_s = self.bottleneck2(f)
            f_t = self.bottleneck(f)
        else:
            f_s = self.bottleneck(f)
            # f_t = self.bottleneck2(f)
            f_t = self.bottleneck(f)
        # f_adv = self.grl_layer(f)
        # outputs_adv = self.adv_head(f_adv)
        outputs = self.head(f_t)
        outputs_pseudo = self.pseudo_head(f_t)
        outputs_source = self.source_head(f_s)
        if save_feat:
            return f, f_s, f_t
        if self.training:
            return outputs, (F.normalize(self.sclhead(f_s),dim=1), F.normalize(self.sclhead(f_t),dim=1)), outputs_pseudo, outputs_source
            # return outputs, None, outputs_pseudo, outputs_source
        else:
            return outputs

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.bottleneck2.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.pseudo_head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.source_head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.sclhead.parameters(), "lr": 1.0 * base_lr},
            {"params": self.adv_head.parameters(), "lr": 1.0 * base_lr}
        ]

        return params

    def step(self):
        self.grl_layer.step()


def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))





class DynamicThresholdingModule(object):
    r"""
    .. math::
        \beta_t(c) = \frac{\sigma_t(c)}{\underset{c'}{\text{max}}~\sigma_t(c')}.

    The dynamic threshold is formulated as

    .. math::
        \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,

    where \tau denotes the pre-defined threshold (e.g. 0.95), :math:`\mathcal{M}` denotes a (possibly non-linear)
    mapping function.

    Args:
        threshold (float): The pre-defined confidence threshold
        warmup (bool): Whether perform threshold warm-up. If True, the number of unlabeled data that have not been
            used will be considered when normalizing :math:`\sigma_t(c)`
        mapping_func (callable): An increasing mapping function. For example, this function can be (1) concave
            :math:`\mathcal{M}(x)=\text{ln}(x+1)/\text{ln}2`, (2) linear :math:`\mathcal{M}(x)=x`,
            and (3) convex :math:`\mathcal{M}(x)=2/2-x`
        num_classes (int): Number of classes
        n_unlabeled_samples (int): Size of the unlabeled dataset
        device (torch.device): Device

    """

    def __init__(self, threshold,  num_classes, n_unlabeled_samples, device, high=0.95, low=0.65):
        self.threshold = torch.ones(num_classes).to(device)*threshold
        self.low = low
        self.high = high
        self.num_classes = num_classes
        self.n_unlabeled_samples = n_unlabeled_samples
        self.device = device
        self.static = None

    def get_threshold(self, confidence, pseudo_labels,warmup=False, lamda=0.9):
        th = self.threshold[pseudo_labels]
        mask = (confidence > th)
        label = pseudo_labels[mask]
        if len(label) == 0:
            self.static = None
            t = torch.ones(self.num_classes).to(self.device)*self.low
            return {'threshold':t,'ratio':-1,'static':-1,'middle':-1,'confidence':confidence,'mask':mask}

        ratio = len(label)/len(pseudo_labels)
        counts = torch.bincount(label.int())
        if len(counts)< self.num_classes:
            counts = torch.cat((counts, torch.zeros(self.num_classes - len(counts)).to(self.device)))
        allcounts = torch.bincount(pseudo_labels.int())
        if len(allcounts)< self.num_classes:
            allcounts = torch.cat((allcounts, torch.zeros(self.num_classes - len(allcounts)).to(self.device)))
        status = [
                counts[c] / allcounts[c] for c in range(len(counts))
            ]
        status = torch.FloatTensor(status).to(self.device)
        status = torch.where(torch.isnan(status),torch.full_like(status,0),status)
        if self.static == None or warmup:
            self.static = status
        else:
            try:
                # self.static = lamda*self.static+(1-lamda)*status
                for i in range(len(self.static)):
                    if allcounts[i]!=0:
                        self.static[i] = lamda*self.static[i]+(1-lamda)*status[i]
            except:
                print(self.static)
        if max(self.static) != 0:
            m = max(self.static)
            status = [
                self.static[c] / m* ratio  for c in range(self.num_classes)
            ]
        middle = status.copy()
        status = [
                max(status[c]*self.high, self.low) for c in range(self.num_classes)
            ]
        status = torch.FloatTensor(status).to(self.device)
        # th = status[pseudo_labels]
        self.threshold = status
        return {'threshold':status,'ratio':ratio,'static':self.static,'middle':middle,'confidence':confidence,'mask':mask}

        # """Calculate and return dynamic threshold"""
        # pseudo_counter = Counter(self.net_outputs.tolist())
        # if max(pseudo_counter.values()) == self.n_unlabeled_samples:
        #     # In the early stage of training, the network does not output pseudo labels with high confidence.
        #     # In this case, the learning status of all categories is simply zero.
        #     status = torch.zeros(self.num_classes).to(self.device)
        # else:
        #     if not self.warmup and -1 in pseudo_counter.keys():
        #         pseudo_counter.pop(-1)
        #     max_num = max(pseudo_counter.values())
        #     # estimate learning status
        #     status = [
        #         pseudo_counter[c] / max_num for c in range(self.num_classes)
        #     ]
        #     status = torch.FloatTensor(status).to(self.device)
        # # calculate dynamic threshold
        # dynamic_threshold = self.threshold * self.mapping_func(status[pseudo_labels])
        # return dynamic_threshold

    def update(self, idxes, selected_mask, pseudo_labels):
        """Update the learning status

        Args:
            idxes (tensor): Indexes of corresponding samples
            selected_mask (tensor): A binary mask, a value of 1 indicates the prediction for this sample will be updated
            pseudo_labels (tensor): Network predictions

        """
        if idxes[selected_mask == 1].nelement() != 0:
            self.net_outputs[idxes[selected_mask == 1]] = pseudo_labels[selected_mask == 1]
