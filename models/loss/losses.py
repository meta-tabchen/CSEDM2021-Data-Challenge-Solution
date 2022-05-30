import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def focal_loss(pred, y, num_classes=2, gamma=2):
    '''
    focal loss

    Parameters:
        pred: 预测标签

        y: 真实标签

        num_classes: 类别数

        gamma

    Return:
        loss: 损失值

    '''
    # focal loss
    target = F.one_hot(y, num_classes=num_classes)
    P = F.softmax(pred, dim=1)
    probs = (P*target).sum(dim=1).view(-1 ,1)  # [batch, 1]
    log_p = probs.log()  # [batch, 1]

    loss = -((torch.pow((1-probs), gamma))*log_p).mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.cross_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = self.cross_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction=='mean':
            return torch.mean(F_loss)
        else:
            return F_loss


def ohem_loss(rate, cls_pred, cls_target, weight, auxiliary_loss=None):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', weight=weight)

    if auxiliary_loss is not None:
        ohem_cls_loss += auxiliary_loss

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


class SupConLoss(nn.Module):
    def __init__(self, nlabels, temperature=0.3, l=0.9):
        """
        《SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINED LANGUAGE MODEL FINE-TUNING》
        nlabels：数据集label总个数
        temperature：
        l：lambda   loss = (1-lambda)*loss_ce + lambda*loss_scl
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.l = l
        self.cross_loss = nn.CrossEntropyLoss()
        self.nlabels = nlabels

    def forward(self, inputs, targets=None):
        batch_size = inputs.shape[0]
        #
        loss_ce = self.cross_loss(inputs, targets)
        #
        N_y = [0 for i in range(self.nlabels)]
        for i in range(batch_size):
            N_y[targets[i]] += 1
        #
        loss_scl = 0
        for i in range(batch_size):
            tmp = 0
            for j in range(batch_size):
                if i == j:
                    continue
                if targets[i] != targets[j]:
                    continue
                #
                numerator = torch.exp(torch.div(torch.matmul(inputs[i],inputs[j]),self.temperature))
                #
                denominator = 0
                for k in range(batch_size):
                    if i == k:
                        continue
                    denominator += torch.exp(torch.div(torch.matmul(inputs[i],inputs[k]),self.temperature))
                #
                tmp += torch.log(torch.div(numerator,denominator))
            #
            if N_y[targets[i]] == 1:  # 防除零
                N_y[targets[i]] = 2
            loss_scl += (((-1) / (N_y[targets[i]] - 1)) * tmp)
        #
        loss = (1-self.l)*loss_ce + self.l*loss_scl
        #
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        loss_ce = self.cross_loss(inputs, targets)
        #
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, dist_ap, y)

        loss = loss_ce + loss_triplet

        return loss