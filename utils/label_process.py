import cv2
import os
from tqdm import tqdm
import numpy as np
root = r'../data/Real/subset/'
sets = ['train','val','test']

def gray2binary(gray):
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    return binary
for st in sets:
    path = os.path.join(root,st,'OUT')
    out_path = os.path.join(root, st, 'OUT_binary')
    os.makedirs(out_path,exist_ok=True)
    names = os.listdir(path)
    for name in tqdm(names):
        gray = cv2.imread(os.path.join(path,name))[:,:,0]
        cv2.imwrite(os.path.join(out_path,name.replace('jpg','png')),gray2binary(gray))
