import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from thop import profile
from utils.utils import visualize_train_ori

"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("tmp/log.txt")
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(seed=777)

train_loader, val_loader = get_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logger.info('LOADING Model')

model = load_model(opt, dev)
input = [torch.randn((2, 3, 256, 256)).cuda()]
flops, params = profile(model, inputs=input)
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=opt.learning_rate)  # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logger.info('STARTING training')
total_step = -1

VIS = 0
if VIS:
    out_path = r'outputs/SiameseCGNet_v6_with_stem_12_31'
    os.makedirs(out_path, exist_ok=True)
    for x in os.listdir(out_path):
        os.remove(os.path.join(out_path, x))
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 4, figsize=(16, 8))

cd_loss = torch.tensor(0.)
bce_loss, dice_loss = torch.tensor(0.), torch.tensor(0.)
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    logger.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels, name in tbar:
        tbar.set_description(
            "epoch {} obj/bg/loss ".format(epoch) + '{:.5f}/{:.5f}/{:.5f}'.format(
                bce_loss.item(), dice_loss.item(),cd_loss.item()))
        batch_iter = batch_iter + opt.batch_size
        total_step += 1
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        if len(labels.shape) == 4:
            labels = labels[:, :, :, 0]
        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds = model(torch.cat([batch_img1, batch_img2],dim=0))
        x_stem, x0_1, x0_2, x0_3 = cd_preds[:4]
        bce_loss, dice_loss, cd_loss = criterion([cd_preds[-1]], labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size ** 2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        if VIS and batch_iter % (opt.batch_size * 200) == 0:
            visualize_train_ori(batch_img1[0], batch_img2[0], labels[0], cd_preds[0],
                                x_stem[0], x0_1[0], x0_2[0], x0_3[0],
                            os.path.join(out_path, name[0].replace('.jpg', '_{}.jpg'.format(epoch))),
                            ax)
        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    logger.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    # validate once every 10 epochs
    if epoch % 10 != 0 and epoch != opt.epochs - 1:
        continue
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels, name in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            if len(labels.shape) == 4:
                labels = labels[:, :, :, 0]
            # Get predictions and calculate loss
            cd_preds = model(torch.cat([batch_img1, batch_img2],dim=0))

            bce_loss, dice_loss, cd_loss = criterion([cd_preds[-1]], labels)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size ** 2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        logger.info("EPOCH {} VALIDATION METRICS".format(epoch) + str(mean_val_metrics))

        """
        Store the weights of good epochs based on validation results
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            # Insert training and epoch information to metadata dictionary
            logger.info('updata the model')
            metadata['validation_metrics'] = mean_val_metrics

            # Save model and log
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, './tmp/checkpoint_epoch_' + str(epoch) + '.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics

        print('An epoch finished.')
writer.close()  # close tensor board
print('Done!')
