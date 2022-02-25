# LSNet
Official implementation of "[LSNet: Extremely lightweight Siamese Network for Change Detection in Remote Sensing Images](https://arxiv.org/abs/2201.09156)".
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
@misc{liu2022lsnet,
    title={LSNet: Extremely Light-Weight Siamese Network For Change Detection in Remote Sensing Image},
    author={Biyuan Liu and Huaixin Chen and Zhixi Wang},
    year={2022},
    eprint={2201.09156},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## References
Note that the source code is implemented with reference to [SNUNet](https://github.com/likyoo/Siam-NestedUNet). 
