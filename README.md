# LSNet
Official implementation of "LSNet: Extremely lightweight Siamese Network for Change Detection in Remote Sensing Images".
[arxiv](https://arxiv.org/abs/2201.09156)|[IGARSS2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9884446)
## Requirements
We use python/pytorch/torchvision versions as follow:  
- python 3.7
- pytorch 1.10.0
- torchvision 0.11.1

You can try a lower version, but the python version is no less than 3.6 and the pytorch version is not less than 1.5. If you have any questions, please submit issue.

## Dataset
We use CDD dataset from [Change Detection in Remote Sensing Images Using Conditional Adversarial Networks](https://paperswithcode.com/paper/change-detection-in-remote-sensing-images)
# Train
For training, you can modify parameters in "metadata.json", or just keep the default and:
```
python train.py
```

# Test
All the pre-trained models have been upload in ./weights, you can modify "model name" in "metadata.json" and
```
python eval.py
```
Noted that the source code has been reconstructed and the results are a little different from the paper. But still keeping efficient. 

## Citation

If you feel it useful, please star and cite our work:
```
@inproceedings{liu2022lsnet,
  title={LSNET: Extremely Light-Weight Siamese Network for Change Detection of Remote Sensing Image},
  author={Liu, Biyuan and Chen, Huaixin and Wang, Zhixi and Xie, Wenqiang and Shuai, LingYu},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={2358--2361},
  year={2022},
  organization={IEEE}
}
```

## References
Note that the source code is implemented with reference to [SNUNet](https://github.com/likyoo/Siam-NestedUNet). 
