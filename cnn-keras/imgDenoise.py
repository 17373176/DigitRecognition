'''图像降噪'''

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
import torchvision


def data_input():
    file = '../data/mnist.npz'
    f = np.load(file)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


# 特征向量二值化
def vec_two_value(data):
    for i in range(data.shape[0]):
        row_data = data[i]
        for j in range(len(row_data)):
            if row_data[j] > 25:  # 多个阈值进行测试，最简单的噪声去除
                data[i][j] = 120
            else:
                data[i][j] = 0
    return data


# 可视化数据集
def display_image(data, img_size):
    plt.figure(figsize=(8, 7))  # 画布大小
    data = data.reshape(img_size, img_size)
    plt.subplot(1, 1, 1)
    plt.imshow(data)
    plt.show()


# main()
X_data, y_data, X_test_data = data_input()  # 读取三个数据文件
generate_file_path = '../data/y_test.xls'
# 将数据变为二维数据 60000 * 28 * 28 -> 60000 * 784
# print(X_data.shape[2])
X_train = np.reshape(X_data, (60000, 28 * 28))
X_test = np.reshape(X_test_data, (10000, 28 * 28))

# 数据可视化

# 训练
# 特征二值化
X_train = vec_two_value(X_train)
X_test = vec_two_value(X_test)
display_image(X_train[2], 28)  # 可视化一张图
display_image(X_data[1], 28)
