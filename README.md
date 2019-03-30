# 68dots_facealignment-master
68个点的人脸特征点检测 train and test
#### 主要 demo 简介：
 +  to_train.py: 训练
 -  test.py: 测试
 +  hg_net.py: <<Stacked Hourglass Networks for Human Pose Estimation>> 论文提出的 HourseglassNet
 +  fy_net.py: 实习公司前辈搭建的网络，原版本是 caffe，被我翻译了一版 pytorch
 +  deal300.py: 数据预处理，由于实际场景需要，把图片转换为黑白
 +  data_loader.py、test_data_loader.py: 数据载入, data_loader.py 与 test_data_loader.py 唯一区别：test_data_loader.py 比data_loader.py 多 return 了 img_name