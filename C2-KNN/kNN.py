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

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines) #get the number of lines in the file
    returnMat=zeros((numberOfLines,3)) # create the return matrix, dim=n*2
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector


