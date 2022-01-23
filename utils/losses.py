import torch.nn
import torch.nn.functional as F
from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss
parser, metadata = get_parser_with_args()
opt = parser.parse_args()
class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin1 = 0.3
        self.margin2 = 2.2
        self.eps = 1e-6
        #self.l1 = torch.nn.L1Loss(reduction='none')

    def forward(self,x1, x2, y):

        dist = torch.functional.norm(x1-x2,dim=1)
        #dist = self.l1(x1,x2).sum(dim=1)
        mdist_bg = torch.clamp(dist-self.margin1, min=0.0)
        mdist_obj = torch.clamp(self.margin2-dist, min=0.0)

        loss_bg = torch.mul(mdist_bg.pow(2),1 - y)
        loss_obj = torch.mul(mdist_obj.pow(2),y)

        return torch.mean(loss_obj) + torch.mean(loss_bg)

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0
    dice= dice_loss()
    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    prediction = predictions[-1]
    bce_l = focal(prediction, target)
    dice_l = dice(prediction, target)
    loss += bce_l + dice_l

    return bce_l,dice_l,loss


class contra_hybrid_loss_align(torch.nn.Module):
    def __init__(self):
        super(contra_hybrid_loss, self).__init__()
        self.contra = ContrastiveLoss()  # torch.nn.L1Loss().cuda()
        self.focal = FocalLoss(gamma=0, alpha=None)# gamma=0, alpha=None --> CE
        self.dice = dice_loss()
    def forward(self, predictions, target):
        """Calculating the loss"""

        xA = torch.mul(predictions[0], predictions[2])
        xB = predictions[1]  # torch.mul(predictions[1], predictions[3])
        # xA_obj = torch.mul(xA, target.unsqueeze(1))
        # xB_obj = torch.mul(xB, target.unsqueeze(1))
        # xA_bg = torch.mul(xA, 1 - target.unsqueeze(1))
        # xB_bg = torch.mul(xB, 1 - target.unsqueeze(1))
        #
        # bg_loss = criteria(xA_bg, xB_bg)
        # obj_loss = 1 - criteria(xA_obj, xB_obj)
        obj_loss, bg_loss = self.contra(xA,xB,target)
        # mask_loss = criteria(predictions[4],target)
        # pred loss
        prediction = predictions[-1]
        #contra_loss = self.contra(prediction,target)
        bce_loss = self.focal(prediction, target)
        dice_loss = self.dice(prediction, target)

        loss = bce_loss + dice_loss + obj_loss + bg_loss

        return obj_loss, bg_loss, bce_loss,loss

class contra_hybrid_loss(torch.nn.Module):
    def __init__(self):
        super(contra_hybrid_loss, self).__init__()
        self.contra = ContrastiveLoss()  # torch.nn.L1Loss().cuda()
        self.focal = FocalLoss(gamma=0, alpha=None)# gamma=0, alpha=None --> CE
        self.dice = dice_loss()
    def forward(self, predictions, target):
        """Calculating the loss"""
        bce_loss = 0
        dice_loss = 0
        # features = predictions[0]
        # for feature_pair in features:
        #     contra_loss += self.contra(feature_pair[0],feature_pair[1],
        #                                F.interpolate(target.unsqueeze(0).float(),size=feature_pair[0].shape[2:])[0].long())
        weight = [1, 1/4, 1/8]
        for idx, prediction in enumerate(predictions):
            bce_loss += weight[idx]*self.focal(prediction,
                                  F.interpolate(target.unsqueeze(0).float(),size=prediction.shape[2:])[0].long())
            dice_loss += weight[idx]*self.dice(prediction,
                                  F.interpolate(target.unsqueeze(0).float(),size=prediction.shape[2:])[0].long())

        loss = bce_loss + dice_loss

        return dice_loss, bce_loss, loss