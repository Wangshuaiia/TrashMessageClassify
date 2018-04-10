#coding=utf-8
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
def similarity(Xi, Xj,sigma):
    length = len(Xi)
    result = 0
    for i in range(length):
        result += (Xi[i] - Xj[i]) ** 2
    Wij = np.exp(- result / (2.0 * (sigma **2)) )
    return Wij
def calculate_W(dataSet,k,sigma):
    '''
    dataSet.shape = (200,2)
    k : 选取k个最邻近的点完成图的构造
    '''
    num = dataSet.shape[0]
    W = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            Xi = dataSet[i]
            Xj = dataSet[j]
            W[i][j] = similarity(Xi, Xj,sigma)
        B = np.argsort(W[i])
        n_index = B[-1:-(k + 1):-1]
        for l in range(num):
            if l not in n_index:
                W[i][l] = 0     #将除k个以外的置零

    return W
def getNormLaplacian(W):
    """input matrix W=(w_ij)
    "compute D=diag(d1,...dn)
    "and L=D-W
    "and Lbar=D^(-1/2)LD^(-1/2)
    "return Lbar
    """
    d = [np.sum(row) for row in W]
    D = np.diag(d)
    L = D - W
    # Dn=D^(-1/2)
    Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
    Lbar = np.dot(np.dot(Dn, L), Dn)
    return Lbar

def getEigVec(L,cluster_num):
    eigval,eigvec = np.linalg.eig(L)
    dim = len(eigval)
    dictEigval = dict(zip(eigval,range(0,dim)))
    kEig = np.sort(eigval)[0:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix],eigvec[:,ix]

def normalize(eigvec):
    row = eigvec.shape[0] #200
    column = eigvec.shape[1] #2
    normVec = np.zeros((row,column))
    for i in range(row):
        add = 0
        for j in range(column):
            add += eigvec[i][j] ** 2
        add = np.sqrt(add)
        for j in range(column):
            normVec[i][j] = eigvec[i][j] / add
    return normVec

def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))
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
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids, clusterAssment

if __name__ == '__main__':
    dataSet = np.zeros((200,2))
    fileIn = open('MoonData.txt')
    lines = fileIn.readlines()
    fileIn.close()
    i = 0
    # for i in range(10):
    for line in lines:
        lineArr = line.strip().split(' ')
        A = [float(lineArr[0]),float(lineArr[1])]
        dataSet[i] = A
        i += 1
    ## step 2: clustering...
    Sigma = np.linspace(0.001,0.08,100)
    i =0
    Accu = []
    n = 30
    # for sigma in Sigma:
    # for step in range(1,n):
    W = calculate_W(dataSet,k=5,sigma=1)
    W = 0.5 * (W.T + W)
    np.savetxt('w.txt', W, fmt='%f', delimiter=' ', newline='\r\n')
    L = getNormLaplacian(W)
    eigval, eigvec = getEigVec(L,2)
    normVec = normalize(eigvec)
    centroids, clusterAssment = kmeans(normVec,2)
    pointsInCluster0 = dataSet[np.nonzero(clusterAssment[:, 0].A == 0)[0]]  #第一类
    pointsInCluster1 = dataSet[np.nonzero(clusterAssment[:, 0].A == 1)[0]]  #第二类
    correctNum = 0
    for predict in pointsInCluster0:
        if predict in dataSet[0:100]:
            correctNum += 1
    for predict in pointsInCluster1:
        if predict in dataSet[100:200]:
            correctNum += 1
    accuracy = correctNum / 200
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    # print(sigma,accuracy)
    Accu.append(accuracy)
    plt.figure(1)
    plt.scatter(pointsInCluster0[:,0],pointsInCluster0[:,1],c='r')
    plt.scatter(pointsInCluster1[:,0],pointsInCluster1[:,1],c='b')
    plt.grid(True)
    # plt.figure(2)
    # # plt.plot(range(1,n),Accu)
    # plt.plot(Sigma,Accu)
    plt.show()