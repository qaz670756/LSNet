import os
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
from utils.utils import visualize_eval_all,feamap_handler
import matplotlib.pyplot as plt

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_loader = get_test_loaders(opt)
model_name = 'LSNet_thin_diffFPN'
path = 'tmp/%s/checkpoint_epoch_100.pt'% model_name  # the path of the model
model = torch.load(path)
model.eval()
# set batch_size to 3 in metadata.json
# if you want set VIS to 1
VIS = 0
if VIS:
    out_path = r'outputs/%s'%model_name
    os.makedirs(out_path, exist_ok=True)
    for x in os.listdir(out_path):
        os.remove(os.path.join(out_path, x))
    f, ax = plt.subplots(3, 5, figsize=(15, 9))
    plt.tight_layout()
    f.tight_layout()
    feamap_h = feamap_handler(model)
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
from thop import profile
input = [torch.randn((2, 3, 256, 256)).cuda()]
flops, params = profile(model, inputs=input)
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, name in tbar:
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        if len(labels.shape) == 4:
            labels = labels[:, :, :, 0]

        cd_preds = model(torch.cat([batch_img1, batch_img2],dim=0))
        feature_list = [cd_preds[0]]
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        if VIS:
            feamaps = feamap_h.show_featuremap(batch_img1, batch_img2, feature_list)
            visualize_eval_all(batch_img1, batch_img2, labels, cd_preds,feamaps,
                            os.path.join(out_path, name[0]),ax)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                          cd_preds.data.cpu().numpy().flatten()).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
OA = (tp+tn)/(tp+tn+fp+fn)

print('Precision: {}\nRecall: {}\nF1-Score: {}\nOA: {}'.format(P, R, F1, OA))
