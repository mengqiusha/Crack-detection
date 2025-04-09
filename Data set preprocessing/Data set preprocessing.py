import cv2
import numpy as np
import os
import random # 提供生成随机数和打乱数据的函数，这里用于打乱数据集。
import h5py

data_directory = "Original image file path"  # 插入将要处理的目录
img_size = 128
categories = ["Positive", "Negative"] # 两类或两类别的图像，可能表明图像是否有裂缝（Positive）或没有裂缝（Negative）。
training_data = [] # 一个空列表，用于存储处理后的图像及其对应的标签。


def create_training_data():
    for category in categories: # 循环遍历“正面”和“负面”类别。
        path = os.path.join(data_directory, category)
        class_num = categories.index(category) # 为类别分配数字标签（0 表示“积极”，1 表示“消极”）。

        # 读取并调整图像大小，并将图像本身及其类号添加到training_data列表
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # 以灰度形式读取图像。
            new_array = cv2.resize(img_array, (img_size, img_size)) # 将图像大小调整为 128x128 像素。
            training_data.append([new_array, class_num]) # 将调整大小后的图像及其对应的类标签附加到training_data列表中。


print("Creating training data...")
create_training_data()
print("Training data successfully created!!")

print("Shuffling training data...")
random.shuffle(training_data) # 对training_data列表进行打乱，使图像和标签的顺序随机化。这有助于防止训练期间出现任何排序偏差。
print("Training data successfully shuffled!!")

X_data = []
y = []

# 用特征（图像）创建X，用目标（标签）创建y
for features, label in training_data: # 循环遍历training_data包含图像标签对的列表。
    X_data.append(features)
    y.append(label)

print("X and y data successfully created!!")

# 将图像重塑为正确的tensorflow格式（图像，宽度，高度，通道）
print("Reshaping X data...")
X = np.array(X_data).reshape(len(X_data), img_size, img_size, 1) # 将列表转换X_data为 NumPy 数组，并将其重塑为形状为 的四维数组(num_samples, img_size, img_size, 1)，这是 TensorFlow 模型所需的格式（图像、高度、宽度、通道）。1表示灰度（单通道）。
print("X data successfully reshaped!!")

print("Saving the data...")

hf = h5py.File(r"File saving path.h5", "w")
hf.create_dataset("X_concrete", data=X, compression="gzip") # 存储图像数据（X），使用 gzip 压缩。
hf.create_dataset("y_concrete", data=y, compression="gzip")
hf.close()
print("Data successfully saved!!")
