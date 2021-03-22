import pandas as pd

f_test = '../data/y_test_data3b=32.xls'
f_val = '../data/val.xls'

test = list(pd.read_excel(f_test)['data'])
val = list(pd.read_excel(f_val)['data'])

accuracy = 0

for i in range(0, len(test)):
    if test[i] is val[i]:
        accuracy = accuracy + 1

print(accuracy/10000)