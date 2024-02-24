def generate(window_size, step):
    '''生成训练样本X和对应标签Y'''
    files = os.listdir("/data/wang_sc/datasets/self_made_dataset")
    X, Y = [], []
    for file in files:
        match = re.match(r'^(\d+)_', file)
        total = np.load("/data/wang_sc/datasets/self_made_dataset/" + file)
        total = list(total)
        label = match.group(1)
        print(f"label is {label}")
        x, y = window(total, label, window_size, step)
        X += x
        Y += y
        
    cate_idx = list(Counter(Y).keys())
    cate_idx.sort()
    for i in range(len(Y)):
        Y[i] = cate_idx.index(Y[i])

    return X, Y

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import os
#索引1为activity_id
#索引[4:16)/[21,33)/[38,50)分别为3个IMU的3D-acc1,3D-acc2,3D-gyro,3D-magn(共36种特征)
loc = [1] + [*range(4,16)] + [*range(21,33)] + [*range(38,50)]

def window(data, label, size, stride):
    '''将数组data和label按照滑窗尺寸size和stride进行切割'''
    x, y = [], []
    for i in range(0, len(label), stride):
        if i+size < len(label): #不足一个滑窗大小的数据丢弃

            l = set(label[i:i+size])
            if len(l) > 1 or label[i] == 0: #当一个滑窗中含有包含多种activity或者activity_id为0（即属于其他动作），丢弃
                continue
            elif len(l) == 1:
                x.append(data[i: i + size, :])
                y.append(label[i])

    return x, y

def generate(window_size, step, is_test, test_subject):
    '''生成训练样本X和对应标签Y'''
    test_user = 2
    files = os.listdir("/data/wang_sc/datasets/self_made_dataset")
    X, Y = [], []
    duration = 0
    for file in files:
        parts = file.split('_')
        if len(parts) > 1:
            user = int(parts[1])
        if is_test and user in test_subject:
            continue
        elif is_test == False and user not in test_subject:
            continue
        total = pd.read_csv("/data/wang_sc/datasets/self_made_dataset/" + file).values
        duration += total.shape[0]
        total = total[~np.isnan(total).any(axis=1), :]  #去除NaN
        data = total[:, 1:10]
        label = total[:, 0].reshape(-1)

        # 调用window函数进行滑窗处理
        x, y = window(data, label, window_size, step)
        X += x
        Y += y
    print(float(duration/100))
    # 将索引从0开始依次编号
    cate_idx = list(Counter(Y).keys())
    cate_idx.sort()
    for i in range(len(Y)):
        Y[i] = cate_idx.index(Y[i])

    return X, Y

def category(X, Y):
    '''按照种类分类动作'''
    result = [[] for i in range(len(list(Counter(Y).keys())))]  #result对应的索引即标签
    for step, y in enumerate(Y):
        result[y].append(X[step])
    return result

def split(result, test_size):
    '''划分数据集
    test_size:测试集样本数量占比'''
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, data in enumerate(result):
        label = [i for n in range(len(data))]
        x_train_, x_test_, y_train_, y_test_ = train_test_split(data, label, test_size=test_size, shuffle=True)
        x_train.extend(x_train_)
        y_train.extend(y_train_)
        x_test.extend(x_test_)
        y_test.extend(y_test_)
    return x_train, y_train, x_test, y_test

def gen_dataset(dataset, test_subject):
    dir_name = dataset + "_subject_" + str(test_subject[0]) + str(test_subject[1])
    dir_path = '/data/wang_sc/datasets/PAMAP2_Dataset/Cross_subject/' + dir_name
    x_train, y_train = generate(171, 85, False, test_subject)
    x_test, y_test = generate(171, 85, True, test_subject)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(dir_path + '/x_train', x_train)
    np.save(dir_path + '/x_test', x_test)
    np.save(dir_path + '/y_train', y_train)
    np.save(dir_path + '/y_test', y_test)

if __name__ == '__main__':
    for i in range(6):
        test_subject = [i % 6, (i+1) % 6]
        print(f"Generating subject {test_subject[0]} and {test_subject[1]}")
        gen_dataset("LIMU", test_subject)