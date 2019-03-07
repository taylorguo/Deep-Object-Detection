## Paper List



2013 

- [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)
    - sliding window detector on an image pyramid




### Single Stage Object Detection




2016

- [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) - ECCV

- YOLOv2 [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

2018

- [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

    - bbox 预测使用尺寸聚类

        - 每个box有4个坐标

        - 训练时, 使用误差平方和损失函数 sum of squared error loss

        - bbox object分值, 用 logistic regression

        - 分类器 使用 logistic regression, 损失函数binary cross-entropy

    - 借鉴了 FPN 网络

    - 特征提取卷积网络

        - 3x3, 1x1 卷积层交替

        - 借鉴了 ResNet, 使用了直连, 分别从卷积层或直连层进行直连




### Multi-stage Object Detection




2014

- RCNN 

    - [Region-Based Convolutional Networks for
    Accurate Object Detection and Segmentation](http://medialab.sjtu.edu.cn/teaching/CV/hw/related_papers/3_detection.pdf)

    - v5 [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524v3.pdf) - CVPR
        - region proposal with scale-normalized before classifying with a ConvNet

- SPPnet [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf) - ECCV


2015

- [Highway Networks](https://arxiv.org/pdf/1505.00387v2.pdf), [中文翻译参考](https://www.cnblogs.com/2008nmj/p/9104744.html)

- [Convolutional Neural Networks at Constrained Time Cost](https://arxiv.org/pdf/1412.1710.pdf)

    - 实验表明: 加深网络, 会出现训练误差

- ResNet [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

    - 残差网络中 Shortcut Connection 参考文章

        - 1995 - [Neural networks for pattern recognition - Bishop]()
        - 1996 - [Pattern recognition and neural networks - Ripley]()
        - 1999 - [Modern applied statistics with s-plus - Venables & Ripley]()

- FCN -[Fully convolutional networks for semantic segmentation](https://arxiv.org/pdf/1411.4038.pdf) - CVPR

- [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf) - ICCV

- RPN(Region Proposal Network) & Anchor Box & [Faster R-CNN: To- wards real-time object detection with region proposal net- works](https://arxiv.org/pdf/1506.01497.pdf) - NIPS

- 基于Faster RCNN物体检索 Faster RCNN Object Search [Faster R-CNN Features for Instance Search](https://arxiv.org/pdf/1604.08893.pdf) 



2016

- ResNet [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) - CVPR


- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)

    - Idea from traditional CV feature pyramids, for compute and memory intensive in DL 

        想法源自传统计算机视觉中的特征金字塔, 深度学习中没用是因为计算密集,占内存

    - bottome-up in FeedForward: deepest layer of each stage should have the strongest features
    
        每阶段的最深的一层应该有最强的特征
