#encoding=utf-8
from time import  time
from gensim.models import Word2Vec
import numpy as np
from sklearn import preprocessing
import math

def get_class_statistic(train_data,trainLable):
    # 计算每一类样本的先验概率P，均值miu，协方差矩阵cov

    (data_num, dimension) = train_data.shape
    if data_num != len(trainLable):
        print('dimension not match')
        return
    count = []
    miu = []
    P = []
    cov = []
    data = []
    for i in range(0,2):
        count.append(0)
        data.append(np.zeros(train_data.shape))

    for i in range(0, data_num):
        data[trainLable[i]][count[trainLable[i]],:] = train_data[i,:]
        count[trainLable[i]] += 1  # 每一类的计数
        # class_index = target_labels.index(train_labels[i, 0])  # 判断第几类，target_labels是自己在主函数中定义的
        # data[class_index][count[class_index], :] = train_data[i, :]  # 用法？ len(train_data[i, :] ) == 339
        # count[class_index] += 1  # 每一类的计数

    for i in range(0,2):
        data[i] = data[i][0: count[i], :]
        P.append(count[i] / float(data_num))  # 先验概率
        miu.append(np.mean(data[i], axis=0))
        cov.append(np.cov(data[i], rowvar=0))

    return P, miu, cov

def dataProcess(testdata_filenumber):
    #训练数据预处理
    train_data_num = 0
    for i in range(1,6):
        if i != testdata_filenumber:
            TraindataFileName = 'file' + str(i) + '.txt'
            file = open(TraindataFileName,'r',encoding='utf-8')
            lines = file.readlines()
            train_data_num += len(lines)
            file.close()

    train_data = np.zeros((train_data_num, 100))
    RowNum = 0
    TrainLabel = []
    for index in range(1,6):
        if index != testdata_filenumber:
            TraindataFileName = 'file' + str(index) + '.txt'
            comfile = open(TraindataFileName, 'r', encoding='utf-8')
            comments = comfile.readlines()
            model = Word2Vec.load('fenci_all_result.model')
            # train_data = np.zeros((10001, 100))
            for sentence in comments:
                i = 0
                words = sentence.strip('\n').split(' ')
                data = np.zeros((len(words), 100))
                label  =  int(words[0])

                if (label == 0) or (label == 1):
                    TrainLabel.append(label)
                else:
                    print('第', RowNum, '行有问题')

                del words[0]  # 第一个元素是标签
                # print(len(words))
                for word in words:
                    # print(word)
                    try:
                        data[i, :] = model[word]
                        i += 1
                        # print(model.most_similar(word))
                    except BaseException:
                        pass
                        # print(RowNum)
                # print(model[u"维系"])
                if (i) != len(words):
                    for j in range(len(words) - i):
                        data = np.delete(data, i, axis=0)
                mean = np.mean(data, axis=0)
                train_data[RowNum, :] = mean
                RowNum += 1

    # 测试数据预处理
    Testfilename = 'file' + str(testdata_filenumber) + '.txt'
    Testcomfile = open(Testfilename, 'r', encoding='utf-8')
    TestComments = Testcomfile.readlines()
    TestRowNum = 0
    TestLabel = []
    test_data = np.zeros((len(TestComments), 100))
    # train_data = np.zeros((10001, 100))
    for sentence in TestComments:
        i = 0
        words = sentence.strip('\n').split(' ')
        data = np.zeros((len(words), 100))
        label = int(words[0])
        if (label == 0) or (label == 1):
            TestLabel.append(label)
        else:
            print('第', TestRowNum, '行有问题')
        del words[0]  # 第一个元素是标签
        # print(len(words))
        for word in words:
            # print(word)
            try:
                data[i, :] = model[word]
                i += 1
                # print(model.most_similar(word))
            except BaseException:
                pass
                # print(TestRowNum)
        # print(model[u"维系"])
        if (i) != len(words):
            for j in range(len(words) - i):
                data = np.delete(data, i, axis=0)
        mean = np.mean(data, axis=0)
        test_data[TestRowNum, :] = mean
        TestRowNum += 1

    return train_data,TrainLabel,test_data, TestLabel


# def TestdataProcess():
#
#
#         # if(RowNum > 10000):
#         #     break
#
#     # 测试数据预处理
#     return

def qdf(test_data, test_labels, P, miu, cov):
    inv_cov = []
    reg = 0.01 * np.eye(cov[0].shape[0], dtype=float)
    for i in range(0, 2):
        inv_cov.append(np.linalg.inv(cov[i] + reg))
    nT = 0
    Wi = []
    wi = []
    wi_0 = []
    for i in range(0, 2):
        a = -0.5 * inv_cov[i]
        b = inv_cov[i].dot(miu[i])
        c = -0.5 * (miu[i].dot(inv_cov[i].dot(miu[i].T))) - 0.5 * np.linalg.det(cov[i]) + math.log(P[i])
        Wi.append(a)
        wi.append(b)
        wi_0.append(c)

    test_num = test_data.shape[0]
    correct_num = 0
    TP = 0
    FP = 0
    for i in range(0, test_num):
        G = []
        for j in range(0, 2):
            g = test_data[i].dot(Wi[j].dot(test_data[i].T)) + test_data[i].dot(wi[j]) + wi_0[j]
            G.append(g)
        prediction = G.index(max(G))
        if testLabel[i] == 1:
            nT += 1  #统计正例个数 （垃圾短信个数）
        if prediction == test_labels[i]:
            correct_num += 1
            if prediction == 1:
                TP += 1
        # if prediction == 1:
        #     print(i)
        if prediction == 1 and prediction != testLabel[i]:
            FP += 1
            # print(i)
    FN = nT - TP
    P = TP / (TP + FP)  #准确率
    R = TP / (TP + FN)  #查全率
    F1 = 2 * P * R / (P + R)
    print('nT',nT,'TP',TP,'FP:',FP)
    print(correct_num/float(test_num), P , R, F1)
    return correct_num/float(test_num), P , R, F1

if __name__ == '__main__':
    Accu = 0
    Pall = 0
    R = 0
    F1 = 0
    d = 0.0 #线上预测时间
    for fileIndex in range(1,6):
        train_data,Label,test_data,testLabel = dataProcess(fileIndex)
        print(train_data.shape)
        print(len(Label))
        P, miu, cov = get_class_statistic(train_data,Label)  #nT nF 是正例和负例个数
        # print(P,miu[0].shape,cov[0].shape)
        d1 = time()
        accuracy, p , r, f1 = qdf(test_data,testLabel,P,miu,cov) #准确率 查全率 F1
        d2 = time()
        d += (d2 - d1)
        print('Singletime:',d)
        print('单独每次的：',accuracy, p , r, f1)
        Accu += accuracy
        Pall += p
        R += r
        F1 += f1
    print(Accu/5.0, Pall/5.0, R/5.0, F1/5.0, d/5.0)
        # print(data)
        # print(data.shape)
        # print(mean)
        # print(mean.shape)
        # break

# model.most_similar(u'全店')