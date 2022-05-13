# Faster R-CNN

## 环境配置：
* Python3.8

* Pytorch1.11.0(未使用GPU)

* pycocotools(Windows:还是额外安装了vs)

* Ubuntu或Centos(不建议Windows)

  

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重

（需要下载后放入backbone文件夹中）

下载地址：

* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是`fasterrcnn_resnet50_fpn_coco.pth`文件，
  不是`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`


## 数据集: PASCAL VOC2012
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 在使用预测脚本时，要将`train_weights`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保验证集或测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights-path`。
