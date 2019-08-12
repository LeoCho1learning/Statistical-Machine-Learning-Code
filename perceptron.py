import numpy as np
import time

def loadData(fileName):
    print('start to read data')
    dataArr = []
    labelArr = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            if int(data[0]) >= 5:
                labelArr.append(1)
            else:
                labelArr.append(0)
            dataArr.append([int(num) / 255 for num in data[1:]])
    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter = 50):
    print('start to trans')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b = 0
    h = 0.0001
    for k in range(iter):
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi
        print('Round %d:%d training' % (k, iter))
    return w, b

def test(dataArr, labelArr, w, b):
    print('start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    count = 0
    for i in range(m):
        yi = labelMat[i]
        xi = dataMat[i]
        result = -1 * yi * (w * xi.T + b)
        if result >= 0:
            count += 1
    print(count)
    print(m)
    accuracy = 1 - (count / m)
    print(accuracy)
    return accuracy

if __name__ == "__main__":
    start = time.time()
    train_data, train_label = loadData('mnist_train.csv')
    test_data, test_label = loadData('mnist_test.csv')
    w, b = perceptron(train_data, train_label, iter = 30)
    accuracy = test(test_data, test_label, w, b)
    end = time.time()
    timeCost = end - start
    print(accuracy, timeCost)