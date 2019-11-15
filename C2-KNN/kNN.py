from numpy import * # was bedeutet *
import operator
from imp import reload
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]) #shape:(4,2)
    labels=['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    :param inX: input vector/sample, need to be classified
    :param dataSet: training data (many samples)
    :param labels: output vector/type of samples
    :param k: k neighbours
    :return: type
    '''
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
        line=line.strip()#截取回车字符
        listFromLine=line.split('\t')# 根据制表符，分割成列表元素
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0) #返回每一列最小值
    maxVals=dataSet.max(0) #返回每一列最大值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    ## tile()~瓦片函数
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat, datingLabels=file2matrix('datingTestSet.txt')
    normMat, ranges, minVals=autoNorm(datingDataMat)
    m=normMat.shape(0) #样本数
    numTestVecs=int(m*hoRatio) #测试集
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:], normMat[numTestVecs:m,:],
                                   datingLabels[numTestVecs:m,3])
        print("the classifier came back with: %d, the real answer is: %d"\
              %(classifierResult,datingLabels[i]))
        if (classifierResult!=datingLabels[i]): errorCount+=1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList=['no at all','in small dose','in large dose']
    #input() like cin>>
    percentTats=float(input(
        "percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0(inArr-minVals/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ", resultList[classifierResult-1])

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList) # the number of training samples
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i] #每个训练样本的名字
        fileStr=fileNameStr.split('.')[0] # filename without filetype
        classNumStr=int(fileStr.split('_')[0]) # in this situation--the real class of each sample
        hwLabels.append(classNumStr) # could use := ???
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr))
        if (classifierResult!=classNumStr): errorCount += 1.0
    print("\nthe total number of error is: %d" %errorCount)
    print("\nthe error rate is: %f" %(errorCount/float(mTest)))

