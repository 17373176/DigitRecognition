""" 基于KNN(K-NearestNeighbor，K近邻算法)的手写数字识别
    机器学习导论，2019-11-15
"""

import numpy as np
import xlwt
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt

# 绘图预处理
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 将数据输出到表格
def print_to_file(data, path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('test')
    ws.write(0, 0, 'data')  # 写入表头
    for i in range(len(data)):
        d = data[i]
        ws.write(i + 1, 0, str(d))
    wb.save(path)


# knn算法，计算欧几里得距离
def knn_procedure(k, test_data, train_data, label_data):
    train_data_size = train_data.shape[0]
    dif = tile(test_data, (train_data_size, 1)) - train_data  # 扩展数组行，并求差值,得到与样本的差值列表
    dis = dif ** 2
    dis = dis.sum(axis=1)  # 行求和
    dis = dis ** 0.5  # 距离
    index_dis = argsort(dis)  # 排序，返回的是索引
    count = {}
    for i in range(0, k):  # 选取距离最小的k个数据
        label = label_data[index_dis[i]]  # 在标签数据中找到当前测试数据所属的类别
        count[label] = count.get(label, 0) + 1  # 统计各类别次数
    sort_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)  # 按照降序排列字典
    return sort_count[0][0]


# 测试集预测
def data_test(x_test, x_data, y_data):
    predict_data = []
    k = 3
    for i in range(x_test.shape[0]):  # 遍历每个测试数据
        predict_data.append(knn(k, x_test[i], x_data, y_data))
    return predict_data


def data_input():
    train_data = np.load("../data/X_train.npz")  # arr_0
    label_data = np.load("../data/y_train.npz")  # arr_0
    test_data = np.load("../data/X_test.npz")  # arr_0
    return train_data['arr_0'], label_data['arr_0'], test_data['arr_0']


# 特征向量二值化
def vec_two_value(data):
    for i in range(data.shape[0]):
        row_data = data[i]
        for j in range(len(row_data)):
            if row_data[j] > 20:  # 多个阈值进行测试，最简单的噪声去除
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


# 数据噪声去除，去除孤立点


# main()
X_data, y_data, X_test_data = data_input()  # 读取三个数据文件
generate_file_path = '../data/y_test_knn.xls'
# 将数据变为二维数据 60000 * 28 * 28 -> 60000 * 784
# print(X_data.shape[2])
X_train = np.reshape(X_data, (60000, 28 * 28))
X_test = np.reshape(X_test_data, (10000, 28 * 28))
print(X_test)
'''

# 特征二值化
X_train = vec_two_value(X_train)
X_test = vec_two_value(X_test)

# 测试
submission_data = data_test(X_test, X_train, y_data)

# 输出到文件
print_to_file(submission_data, generate_file_path)

'''