# YOLOv3

## 数据预处理
* 首先下载Pascal VOC2012数据集，将VOCdevkit文件夹与`change.py`置于同一目录下，运行`change.py`，将VOC数据集转为yolo支持的格式。
* 将得到的`images`与`labels`文件夹置于`vocdata`文件夹中，放在根目录下。

## 训练
* 修改`data`文件夹下`voc_test.yaml`中train与val的路径。
* 运行根目录下的`train.py`文件。
* 在`runs\train`文件夹下可查看训练结果。

## 测试
* 将根目录下的`detect.py`文件第206行的路径修改为训练后所得权重的路径，运行`detect.py`，在`runs\train`文件夹下可查看测试结果。