# Deep-Object-Detection
Inspired by awesome object detection, deep object detection does a easy way for understanding in Chinese.

## 主要的网络架构

<img src="./assets/block_diagram/lenet_alexnet.png" width="600">

### 代码实现

[LeNet-Keras for mnist handwriting digital image classification](https://github.com/taylorguo/Deep-Object-Detection/blob/master/sample-code/network/lenet_keras.py)

LeNet-Keras restructure

<img src="./assets/code_diagram/lenet_revised.png" width="500">
Accuracy: 98.54%


===================================

[AlexNet-Keras for oxflower17 image classification](https://github.com/taylorguo/Deep-Object-Detection/blob/master/sample-code/network/alexnet_keras.py)

AlexNet-Keras restructure: 修改后的网络 val_acc: ~80%, 过拟合

<img src="./assets/code_diagram/alexnet_revised_v1.png" width="400">


===================================

<img src="./assets/block_diagram/vgg16.png" width="800">

[VGG16 Keras 官方代码实现](https://github.com/taylorguo/Deep-Object-Detection/blob/master/sample-code/network/vgg16.py)

[VGG16-Keras oxflower17 物体分类](https://github.com/taylorguo/Deep-Object-Detection/blob/master/sample-code/network/vgg16_keras.py): 修改后的网络 val_acc: ~86.4%, 过拟合

<img src="./assets/code_diagram/vgg16_tl.png" width="400">


<img src="./assets/block_diagram/vgg19.png" width="800">

[VGG19 Keras 官方代码实现](https://github.com/taylorguo/Deep-Object-Detection/blob/master/sample-code/network/vgg19.py)
