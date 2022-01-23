from functools import partial
from utils.fmap_visualize import FeatureMapVis_siamese,FeatureMapVis
from pathlib import Path
from utils.fmap_visualize import show_tensor, show_img
import numpy as np
import os
import cv2
import torch
import torchvision
#from models.Models_tmp import SiameseCGNet_v2

def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    outs = self.forward(img)
    return outs


def create_featuremap_vis(model=None, use_gpu=True, init_shape=(768, 768, 3)):
    #model.forward = partial(forward, model)
    featurevis = FeatureMapVis_siamese(model, use_gpu)
    featurevis.set_hook_style(init_shape[2], init_shape[:2])
    return featurevis


def _show_save_data(featurevis, input, img_orig, feature_indexs, filepath, is_show, output_dir):
    show_datas = []
    for feature_index in feature_indexs:
        feature_map = featurevis.run(input, feature_index=feature_index)[0]
        data = show_tensor(feature_map[0], resize_hw=input.shape[:2], show_split=False, is_show=False)[0]
        am_data = cv2.addWeighted(data, 0.5, img_orig, 0.5, 0)
        show_datas.append(am_data)
    if is_show:
        show_img(show_datas)
    if output_dir is not None:
        filename = os.path.join(output_dir,
                                Path(filepath).name
                                )
        if len(show_datas) == 1:
            cv2.imwrite(show_datas[0], filename)
        else:
            for i in range(len(show_datas)):
                fname, suffix = os.path.splitext(filename)
                cv2.imwrite(fname + '_{}'.format(str(i)) + suffix,show_datas[i])

def _show_save_data_siamese(featurevis, input, img_orig, feature_indexs, fmaps = None):
    show_datas = []
    imgs = img_orig.detach().cpu().numpy().transpose([0, 2, 3, 1])
    if fmaps is None:
        for feature_index in feature_indexs:
            feature_maps = featurevis.run(input, feature_index=feature_index)[0]
            am_data = []
            for img,feature_map in zip(imgs,feature_maps[:3]):
                data = show_tensor(feature_map, resize_hw=input.shape[2:], show_split=False, is_show=False)[0]
                am_data.append(255-data)
            show_datas.append(am_data)
    else:
        for i in range(len(fmaps)):
            am_data = []
            for img, feature_map in zip(imgs, fmaps[i]):
                data = show_tensor(feature_map, resize_hw=input.shape[2:], show_split=False, is_show=False)[0]
                am_data.append(data)
            show_datas.append(am_data)
    return show_datas

def show_featuremap_from_imgs(featurevis, feature_indexs, img_dir, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    img_paths = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if 'jpg' in x]
    for path in img_paths:
        img = cv2.imread(path)
        # 这里是输入模型前的图片处理
        input = img.astype(np.float32).copy()
        # 显示特征图
        _show_save_data(featurevis, input, img, feature_indexs, path, is_show, output_dir)



if __name__ == '__main__':
    img_dir = '.'
    out_dir = './out'

    init_shape = (1024, 1024, 3)  # 值不重要，只要前向一遍网络时候不报错即可
    feature_index = [32, 70, 96, 155]  # 可视化的层索引
    use_gpu = True
    is_show = False
    model = torchvision.models.resnet50(pretrained=True)  # 这里创建模型
    #model = torch.load(path)
    if use_gpu:
        model.cuda()
    featurevis = create_featuremap_vis(model, use_gpu, init_shape)
    show_featuremap_from_imgs(featurevis, feature_index, img_dir, is_show, out_dir)
