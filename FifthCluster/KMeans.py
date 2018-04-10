#coding=utf-8
import matplotlib.pyplot as plt
from numpy import random
import numpy as np

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

def LoadData():
    X = np.zeros((1000,2))
    mean1=[1,-1]
    cov=[[1,0],[0,1]]
    x1=random.multivariate_normal(mean1,cov,200)
    # print(x1.shape)
    X[0:200] = x1
    mean2=[5.5,-4.5]
    x2=random.multivariate_normal(mean2,cov,200)
    X[200:400] = x2
    mean3=[1,4]
    x3=random.multivariate_normal(mean3,cov,200)
    X[400:600] = x3
    mean4=[6,4.5]
    x4=random.multivariate_normal(mean4,cov,200)
    X[600:800] = x4
    mean5=[9,0.0]
    x5=random.multivariate_normal(mean5,cov,200)
    X[800:1000] = x5

    # plt.figure(1)  # 创建图表1
    # plt.scatter(x1[:,0],x1[:,1],c='red')
    # plt.scatter(x2[:,0],x2[:,1],c='blue')
    # plt.scatter(x3[:,0],x3[:,1],c='k')
    # plt.scatter(x4[:,0],x4[:,1],c='green')
    # plt.scatter(x5[:,0],x5[:,1],c='m')
    # # plt.scatter(X[:,0],X[:,1])
    # # plt.grid(True)
    # plt.show()

    # print(X.shape)
    return  X

# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    # clusterAssment = np.mat(np.zeros((numSamples)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)
    print(clusterAssment.shape)
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0   #最小值属于哪一类的索引
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            print('第',j+1,'类个数：',pointsInCluster.shape[0])
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids, clusterAssment

if __name__ == '__main__':
    data = LoadData()
    centroids, clusterAssment = kmeans(data, 5)
    for i in range(5):
        print('第',i+1,'类样本聚类中心',centroids[i])
    centers = np.array([[1.0,-1],[5.5,-4.5],[1.0,4],[6,4.5],[9,0.0]])
    dis = 0.0
    for center in centers:
        distance = np.array([99.0,99.0,99.0,99.0,99.0])
        i = 0
        for predict in centroids:
            distance[i] = euclDistance(center,predict)
            # print(distance[i])
            i += 1
        dis += np.min(distance)
        # print(np.where(np.min(distance)))
    # dis = dis / 5.0
    print('均方误差：',dis)  #聚类中心均方误差
    # print(clusterAssment[750:850])