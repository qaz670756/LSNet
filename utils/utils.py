import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from skimage import exposure
from utils.fmap_visualize.fmap_analyze_one_file import _show_save_data_siamese,create_featuremap_vis
import cv2

class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)

def normMask(mask, strenth = 0.5):
    """
    :return: to attention more region
    """
    c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(1,-1).max(1)[0]
    max_value = max_value.reshape(1, 1, 1)
    mask = mask/(max_value*strenth)
    mask = torch.clamp(mask, 0, 1)

    return mask

class feamap_handler():
    def __init__(self,model):
        init_shape = (256, 256, 3)
        self.feature_index = [199,166,66,107]  # 可视化的层索引
        if not isinstance(self.feature_index, (list, tuple)):
            self.feature_index = [self.feature_index]
        use_gpu = True
        self.featurevis = create_featuremap_vis(model, use_gpu, init_shape)
    def show_featuremap(self, xAs,xBs,fmaps=None):
        input = torch.cat([xAs, xBs], dim=0)
        if fmaps is None:
            return _show_save_data_siamese(self.featurevis, input, xAs, self.feature_index)
        else:
            return _show_save_data_siamese(self.featurevis, input, xAs, self.feature_index, fmaps=fmaps)

def toArray(x):
    if len(x.shape) == 3:
        if x.shape[0]>3:
            x = x[:3,:,:]
        out = x.permute((1, 2, 0)).cpu().detach().numpy()
        out = np.uint8((out - np.min(out)) / (np.max(out) - np.min(out) + 1e-8) * 255)
    else:
        out = x.permute((0, 1)).cpu().detach().numpy()
        out = np.uint8(out * 255)
    return out

def visualize_eval(x, y, gt, pred, img_path,ax):
    fontsize = 18
    #f, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(toArray(x))
    ax[0, 0].set_title('Image T1', fontsize=fontsize)

    ax[0, 1].imshow(toArray(y))
    ax[0, 1].set_title('Image T2', fontsize=fontsize)

    ax[1, 0].imshow(toArray(gt),cmap='gray')
    ax[1, 0].set_title('GT', fontsize=fontsize)

    # ax[1, 1].imshow(np.abs(toArray(y)[...,::-1]-toArray(out)[...,::-1]))
    # ax[1, 1].set_title('Difference Image', fontsize=fontsize)
    ax[1, 1].imshow(toArray(pred) ,cmap='gray')
    ax[1, 1].set_title('Pred', fontsize=fontsize)

    plt.savefig(img_path)
def visualize_eval_all(x, y, gt, pred, fmaps, img_path,ax):

    plt.tight_layout()
    #f.tight_layout()
    for idx in range(ax.shape[0]):

        ax[idx, 0].imshow(toArray(x[idx]))
        ax[idx, 0].set_xticks([])
        ax[idx, 0].set_yticks([])

        ax[idx, 1].imshow(toArray(y[idx]))
        ax[idx, 1].set_xticks([])
        ax[idx, 1].set_yticks([])

        ax[idx, 2].imshow(toArray(gt[idx]),cmap='gray')
        ax[idx, 2].set_xticks([])
        ax[idx, 2].set_yticks([])

        ax[idx, 3].imshow(toArray(pred[idx]),cmap='gray')
        ax[idx, 3].set_xticks([])
        ax[idx, 3].set_yticks([])

        ax[idx, 4].imshow(fmaps[0][idx],cmap=plt.cm.jet)
        ax[idx, 4].set_xticks([])
        ax[idx, 4].set_yticks([])
    labels = ['(a)','(b)','(c)','(d)','(e)']
    for i in range(ax.shape[1]):
        ax[idx,i].set_xlabel(labels[i],fontsize=30)
        #ax[idx, 5].imshow(fmaps[1][idx],cmap=plt.cm.jet)
        #ax[idx, 6].imshow(fmaps[2][idx],cmap=plt.cm.jet)


    # for i in range(len(fmaps)):
    #     cv2.imwrite(img_path.replace('.jpg','_%d.jpg'%i), fmaps[i][0])

    plt.savefig(img_path,dpi=200,pad_inches=0.1,bbox_inches='tight')
def visualize_eval_multi(x, y, gt, pred0, pred1, pred2, img_path,ax):
    fontsize = 18


    ax[0, 0].imshow(toArray(x))
    ax[0, 0].set_title('Image T1', fontsize=fontsize)

    ax[0, 1].imshow(toArray(y))
    ax[0, 1].set_title('Image T2', fontsize=fontsize)

    ax[0, 2].imshow(toArray(gt),cmap='gray')
    ax[0, 2].set_title('GT', fontsize=fontsize)

    # ax[1, 1].imshow(np.abs(toArray(y)[...,::-1]-toArray(out)[...,::-1]))
    # ax[1, 1].set_title('Difference Image', fontsize=fontsize)
    ax[1, 0].imshow(toArray(pred0) ,cmap='gray')
    ax[1, 0].set_title('Pred0', fontsize=fontsize)

    ax[1, 1].imshow(toArray(pred1) ,cmap='gray')
    ax[1, 1].set_title('Pred1', fontsize=fontsize)

    ax[1, 2].imshow(toArray(pred2) ,cmap='gray')
    ax[1, 2].set_title('Pred2', fontsize=fontsize)

    plt.savefig(img_path)


def visualize_train(x, y, gt, pred, fA, fB, xA, xB, mask_xA, mask_xB, img_path, ax):
    fontsize = 18

    ax[0, 0].imshow(toArray(x))
    ax[0, 0].set_title('Image T1', fontsize=fontsize)

    ax[0, 1].imshow(toArray(y))
    ax[0, 1].set_title('Image T2', fontsize=fontsize)

    ax[0, 2].imshow(toArray(gt), cmap='gray')
    ax[0, 2].set_title('GT', fontsize=fontsize)

    ax[0, 3].imshow(toArray(pred), cmap='gray')
    ax[0, 3].set_title('Pred', fontsize=fontsize)

    ax[1, 0].imshow(toArray(fA), cmap='gray')
    ax[1, 0].set_title('fA', fontsize=fontsize)

    ax[1, 1].imshow(toArray(fB), cmap='gray')
    ax[1, 1].set_title('fB', fontsize=fontsize)

    ax[1, 2].imshow(toArray(xA), cmap='gray')
    ax[1, 2].set_title('xA', fontsize=fontsize)

    ax[1, 3].imshow(toArray(xB), cmap='gray')
    ax[1, 3].set_title('xB', fontsize=fontsize)

    ax[2, 0].imshow(toArray(mask_xA), cmap='gray')
    ax[2, 0].set_title('Mask_xA', fontsize=fontsize)

    ax[2, 1].imshow(toArray(mask_xB), cmap='gray')
    ax[2, 1].set_title('Mask_xB', fontsize=fontsize)

    ax[2, 2].imshow(exposure.equalize_hist(toArray(xA)), cmap='gray')
    ax[2, 2].set_title('hist_xA', fontsize=fontsize)

    ax[2, 3].imshow(exposure.equalize_hist(toArray(xB)), cmap='gray')
    ax[2, 3].set_title('hist_xB', fontsize=fontsize)

    plt.savefig(img_path)
def visualize_train_ori(x, y, gt, pred, x_stem, x0_1, x0_2, x0_3, img_path, ax):
    fontsize = 18

    ax[0, 0].imshow(toArray(x))
    ax[0, 0].set_title('Image T1', fontsize=fontsize)

    ax[0, 1].imshow(toArray(y))
    ax[0, 1].set_title('Image T2', fontsize=fontsize)

    ax[0, 2].imshow(toArray(gt), cmap='gray')
    ax[0, 2].set_title('GT', fontsize=fontsize)

    ax[0, 3].imshow(toArray(pred), cmap='gray')
    ax[0, 3].set_title('Pred', fontsize=fontsize)

    ax[1, 0].imshow(toArray(x_stem), cmap='gray')
    ax[1, 0].set_title('x_stem', fontsize=fontsize)

    ax[1, 1].imshow(toArray(x0_1), cmap='gray')
    ax[1, 1].set_title('x0_1', fontsize=fontsize)

    ax[1, 2].imshow(toArray(x0_2), cmap='gray')
    ax[1, 2].set_title('x0_2', fontsize=fontsize)

    ax[1, 3].imshow(toArray(x0_3), cmap='gray')
    ax[1, 3].set_title('x0_3', fontsize=fontsize)


    plt.savefig(img_path)

def visualize_tsne_sshape():
    """对S型曲线数据的降维和可视化"""
    # 生成1000个S型曲线数据
    x, color = datasets.make_s_curve(n_samples=1000,
                                     random_state=0)  # x是[1000,2]的2维数据，color是[1000,1]的一维数据

    n_neighbors = 10
    n_components = 2

    # 创建自定义图像
    fig = plt.figure(figsize=(8, 8))  # 指定图像的宽和高
    plt.suptitle("Dimensionality Reduction and Visualization of S-Curve Data ", fontsize=14)  # 自定义图像名称

    # 绘制S型曲线的3D图像
    ax = fig.add_subplot(211, projection='3d')  # 创建子图
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
    ax.set_title('Original S-Curve', fontsize=14)
    ax.view_init(4, -72)  # 初始化视角

    # t-SNE的降维与可视化
    ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(x)
    ax1 = fig.add_subplot(2, 1, 2)
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.set_title('t-SNE Curve', fontsize=14)
    # 显示图像
    plt.show()


# 加载数据
def get_data():
    """
    :return: 数据集、标签、样本数量、特征数量
    """
    digits = datasets.load_digits(n_class=10)
    data = digits.data  # 图片特征
    label = digits.target  # 图片标签
    n_samples, n_features = data.shape  # 数据集的形状
    return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def visualize_tsne():
    data, label, n_samples, n_features = get_data()  # 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    result = ts.fit_transform(data)
    # 调用函数，绘制图像
    # abel is used for coloring
    # result -> [N,2] means [N,(x,y)]
    fig = plot_embedding(result, label, 't-SNE Embedding of digits')
    # 显示图像
    plt.show()


if __name__ == '__main__':
    visualize_tsne()
