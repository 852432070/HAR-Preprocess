import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

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

def generate(window_size, step):
    '''生成训练样本X和对应标签Y'''
    X, Y = [], []
    # 遍历9个subject文件
    for i in range(1, 10):
        total = pd.read_csv('/data/wang_sc/datasets/PAMAP2_Dataset/Protocol/subject10' + str(i) + '.dat', header=None, sep=' ', usecols=loc).values
        total = total[~np.isnan(total).any(axis=1), :]  #去除NaN
        # data = total[:, 1:]
        data = total[:, 8:17]
        label = total[:, 0].reshape(-1)

        # 调用window函数进行滑窗处理
        x, y = window(data, label, window_size, step)
        X += x
        Y += y

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

from scipy.signal import butter, filtfilt

def lowpass_filter(signal_frames, cutoff_freq, sample_rate):
    # 定义低通滤波器参数
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(4, normalized_cutoff_freq, btype='low', analog=False)

    # 对每个信号帧进行滤波
    filtered_frames = np.zeros_like(signal_frames)
    num_frames = signal_frames.shape[0]
    for i in range(num_frames):
        for j in range(signal_frames.shape[2]):
            filtered_frames[i, :, j] = filtfilt(b, a, signal_frames[i, :, j])

    return filtered_frames

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

if __name__ == '__main__':
    X, Y = generate(171, 85)
    result = category(X, Y)
    x_train, y_train, x_test, y_test = split(result, 0.2)

    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed1/x_train', x_train)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed1/x_test', x_test)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed1/y_train', y_train)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed1/y_test', y_test)

    cutoff_freq = 10  # 低通截止频率，单位为Hz
    sample_rate = 100  # 采样率，单位为Hz

    x_train = lowpass_filter(np.array(x_train), cutoff_freq, sample_rate)
    x_test = lowpass_filter(np.array(x_test), cutoff_freq, sample_rate)
    print(x_train.shape)
    print(x_test.shape)

    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed_filter10/x_train', x_train)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed_filter10/x_test', x_test)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed_filter10/y_train', y_train)
    np.save('/data/wang_sc/datasets/PAMAP2_Dataset/Processed_filter10/y_test', y_test)
