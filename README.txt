
公开裂缝数据集：
共40000张图片，包含裂缝图片20000张，非裂缝图片20000张。
数据集下载地址：https://data.mendeley.com/datasets/5y9wdsg2zt/1
这些图像涵盖了多种类型的裂缝，包括混凝土、路面和墙面裂缝。
数据集预处理代码见[Data set preprocessing.py]

裂缝识别：
使用改进的EfficientNet-B0模型结合ECA注意机制，模型代码见[Improved efficientnet_b0 model training.py]
该模型架构有良好的裂缝检测性能，保持了计算效率和检测可靠性。
模型评估指标代码见[Model evaluation.py]

裂缝分割:
改进的EfficientNet-B0模型有高效的网络结构和较低的计算量，作为特征提取的主干网络与Mask R-CNN级联，并结合算法提升裂缝图片分割精度，与裂缝标签计算IOU值
分割模型评估IOU代码见[Segmentation model to evaluate IOU.py]

裂缝量化：
使用连通域检测、骨架提取、形态学等算法计算裂缝长度和平均宽度，最后进行裂缝量化误差分析。

