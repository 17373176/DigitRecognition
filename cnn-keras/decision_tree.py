""" 决策树分析处理文件，对训练集和标签集进行训练，根据测试集生成最终数据
"""

# # decisionTree决策树——对应接口，分析dengue登革热与气候环境数据，预测某地未来的发病案例 # #

from sklearn import tree  # 决策树
from sklearn.tree import export_graphviz  # 将训练好的决策树模型可视化,使用export_graphviz()方法，通过生成一个叫做.dot的图形定义文件
import xlwt
from matplotlib import pyplot as plt
import numpy as np

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


# 训练得到决策树
def deci_tree_train(X, y):
    tree_clf = tree.DecisionTreeClassifier(
        criterion='gini', splitter='best', max_depth=18, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, class_weight=None, presort=False
    )  # 决策树
    tree_clf = tree_clf.fit(X, y)  # 训练拟合
    return tree_clf


# 决策树可视化生成gv文件
def deci_tree_visual(tree_clf):
    dot_data = export_graphviz(
        tree_clf,
        out_file='../data/dot.dot',
        rounded=True,
        filled=True
    )  # 根据决策树导出得到dot文件
    '''graph = graphviz.Source(dot_data)  # 将dot文件转化成gv文件
    graph.render('deci_tree.gv', directory='../data/', view=True)'''


# 根据决策树和测试集得到相应的预测数据
def generator(deci_tree, x_test):
    predict_data = deci_tree.predict(x_test)
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
            if row_data[j] > 45:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


# main()
X_data, y_data, X_test_data = data_input()  # 读取三个数据文件
generate_file_path = '../data/y_test.xls'

# 将数据变为二维数据 60000 * 28 * 28 -> 60000 * 784
# print(X_data.shape[2])
X_train = np.reshape(X_data, (60000, 28 * 28))
X_test = np.reshape(X_test_data, (10000, 28 * 28))

#print_to_file(X_data, '../data/x_train.xls')
#print_to_file(y_data, '../data/y_train.xls')
#print_to_file(X_test_data, '../data/x_test.xls')
# 训练
# 特征二值化
X_train = vec_two_value(X_train)
X_test = vec_two_value(X_test)

my_tree = deci_tree_train(X_train, y_data)

# 决策树可视化
deci_tree_visual(my_tree)

# 根据决策树和测试集生成预测数据
submission_data = generator(my_tree, X_test)

# 输出到文件
print_to_file(submission_data, generate_file_path)
