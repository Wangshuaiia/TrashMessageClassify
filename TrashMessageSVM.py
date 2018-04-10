from sklearn import svm
#encoding=utf-8
from time import  time
from gensim.models import Word2Vec
import numpy as np
from sklearn import preprocessing
import math
from sklearn.externals import joblib

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

def dataProcessOLD():
    #训练数据预处理
    comfile = open('D:\python\homework\WebDataMining\FenciResult1.txt', 'r', encoding='utf-8')
    comments = comfile.readlines()
    model = Word2Vec.load('result.model')
    RowNum = 0
    TrainLabel = []
    train_data = np.zeros((len(comments), 100))
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
    Testcomfile = open('D:\python\homework\WebDataMining\FenciResult2.txt', 'r', encoding='utf-8')
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

if __name__ == '__main__':
    Accu = 0
    Pall = 0
    Rall = 0
    F1all = 0
    d = 0.0 #线上预测时间
    for fileIndex in range(1, 6):
        train_data,Label,test_data,testLabel = dataProcess(fileIndex)
        # train_data,Label,test_data,testLabel = dataProcessOLD()
        print(len(testLabel))
        print(test_data.shape)
        # clf = svm.SVC(max_iter=100000)  # class
        # clf.fit(train_data, Label)  # training the svc model
        clf = joblib.load("SVM_model_1.m")  #有训练模型以后，用这句话来调用训练模型
        # joblib.dump(clf, "SVM_model_1.m")  #存储训练模型
        print("begin predict")
        d1 = time()
        result = clf.predict(test_data)
        d2 = time()
        d += d2 - d1
        correctNum = 0
        nT = 0   #垃圾短信个数
        TP = 0
        FP = 0
        for i in range(len(testLabel)):
            print(testLabel[i],result[i])
            if testLabel[i] == 1:
                nT += 1         #正例个数
            if result[i] == testLabel[i]:
                correctNum += 1
                if result[i] == 1:
                    TP += 1   #真正例个数
            if result[i] == 1 and result[i] != testLabel[i]:
                FP += 1  #假正例个数
        print(TP,FP,nT,correctNum)
        FN = nT - TP
        P = TP / (TP + FP)  # 准确率
        R = TP / (TP + FN)  # 查全率
        F1 = 2 * P * R / (P + R)
        accuracy = correctNum/float(len(testLabel))
        Accu += accuracy
        Pall += P
        Rall += R
        F1all += F1
        print(accuracy)  #accuracy
        print('P:',P)
        print('R:',R)
        print('正常短信误判为垃圾短信的个数：',FP)
        print('F1:',F1)
    print(Accu/5.0, Pall/5.0, Rall/5.0, F1all/5.0, d/5.0)




