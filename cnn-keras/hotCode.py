from keras.utils import np_utils
import xlwt
import pandas as pd


# 将数据输出到表格
def print_to_file(data, path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('test')
    ws.write(0, 0, 'data')  # 写入表头
    for i in range(data.shape[0]):
        row_data = data[i]
        k = 0
        for j in range(len(row_data)):
            if row_data[j] == 1:
                ws.write(i + 1, 0, str(j))
                k = 1
            if k == 1:
                break
    wb.save(path)


data = pd.read_excel('../data/y_test.xls')

# data = np_utils.to_categorical(data, 10)

print_to_file(data, '../data/y_testOK.xls')  # 转化独热编码为原数据
