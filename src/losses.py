from torch import nn
import segmentation_models_pytorch as smp


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
FocalLoss = smp.losses.FocalLoss(mode='multilabel')

def criterion_seg(y_pred, y_true, alpha=0.01, beta=0.99, loss_weights=[1,1]):
    # bs, n, c, h, w = y_true.shape[:2]
    # y_pred = y_pred.view(bs*n, c, h, w)
    # y_true = y_true.view(bs*n, c, h, w)
    CELoss      = nn.CrossEntropyLoss()#smp.losses.SoftCrossEntropyLoss()# [0,1,0,2,1,2,1]
    TverskyLoss = smp.losses.TverskyLoss(mode='multiclass', classes=[0,1,3,4,5,6], alpha=alpha, beta=beta, log_loss=False)
    bce = CELoss(y_pred, y_true) * loss_weights[0]
    tvs = TverskyLoss(y_pred, y_true) * loss_weights[1]
    return bce + tvs, bce, tvs