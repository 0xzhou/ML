from numpy import * # was bedeutet *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]) #shape:(4,2)
    labels=['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #-----------Calculation of the distance----------------------
    dataSetSize=dataSet.shape[0] #number of samples in TraingSet
    diffMat=tile(inX, (dataSetSize,1))-dataSet # difference vector of inX with each samples
    sqDiffMat=diffMat**2 # square based on each elements
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #------------sort-----------------------
    sortedDistIndicies=distances.argsort()# indices!! from small to big
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 # count the k nearst elements based on labels
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]