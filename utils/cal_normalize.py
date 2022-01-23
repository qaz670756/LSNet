import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

img_h, img_w = 256, 256
means, stdevs = [], []
img_list = []

TRAIN_DATASET_PATH = 'data/Real/subset/train/B'

image_fns = glob(os.path.join(TRAIN_DATASET_PATH, '*.*'))

for single_img_path in tqdm(image_fns):
    img = cv2.imread(single_img_path)
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)


imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))

# normMean = [0.35389897, 0.39104056, 0.34307468]
# normStd = [0.2158508, 0.23398565, 0.20874721]
# normMean1 = [0.47324282, 0.498616, 0.46873462]
# normStd1 = [0.2431127, 0.2601882, 0.25678185]
# [0.413570895, 0.44482827999999996, 0.40590465]
# [0.22948174999999998, 0.24708692499999999, 0.23276452999999997]