from __future__ import division
import sys
import os
import glob
import pickle
#import nltk.data
import numpy as np
from keras.preprocessing.text import Tokenizer


def genPatientList(labelledList, idList):
    patientDict = {}
    allPatientId = []
    for currentId in idList:
        currentPatientId = labelledList[currentId][0]
        if currentPatientId in patientDict:
            patientDict[currentPatientId].append(currentId)
        else:
            patientDict[currentPatientId] = [currentId]
    return patientDict

def get4classID(npItem):
    idx = int(np.where(npItem == max(npItem))[0])
    return idx

def get2classID(npItem):
    value = npItem[0]
    #print(value)
    if value > 0.5:
        idx = 1
    else:
        idx = 0
    return idx



def evaluation(avilableLabels, prediction, y, onehot=True):
    print(avilableLabels)
    correct = 0
    incorrect = 0
    evaluationListRecall=[[] for _ in range(len(avilableLabels))]
    evaluationListPrecision=[[] for _ in range(len(avilableLabels))]
    evaluationListAccuracy=[[] for _ in range(len(avilableLabels))]
    for j in range(len(evaluationListRecall)):
        evaluationListRecall[j] = [0.01,0.01] #[correct, incorrect]
        evaluationListPrecision[j] = [0.01,0.01]
        evaluationListAccuracy[j] = [0.01,0.01]
    i=0
    for item in prediction:
        if len(avilableLabels) > 2:
            if onehot == True:
                yidx = get4classID(y[i])
                pidx = get4classID(item)
            else:
                pidx = item
                yidx = y[i]
        else:
            if onehot == True:
                yidx = get2classID(y[i])
                pidx = get2classID(item)
            else:
                pidx = item
                yidx = y[i]
        #print pidx, yidx
        if pidx == yidx:
            correct += 1
            evaluationListRecall[pidx][0] += 1
            evaluationListPrecision[pidx][0] += 1
        else:
            incorrect += 1
            evaluationListPrecision[pidx][1] += 1
            evaluationListRecall[yidx][1] += 1
        i+=1
    print(evaluationListRecall, evaluationListPrecision)
    accuracy = correct/(correct+incorrect)
    print('accuracy ', accuracy)
    for i in range(len(avilableLabels)):
        print(avilableLabels[i])
        recall =  evaluationListRecall[i][0]/ (evaluationListRecall[i][0]+evaluationListRecall[i][1])
        precision = evaluationListPrecision[i][0] / (evaluationListPrecision[i][0]+ evaluationListPrecision[i][1])
        fmeasure = 2*(recall * precision)/(recall+precision)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f-measure ', fmeasure)


def transSent2DocLabel(avilableLabels, labelList):
    if 'Suicidality' in labelList:
        docLabel = avilableLabels.index('Suicidality')
    elif 'Uncertain' in labelList:
        docLabel = avilableLabels.index('Uncertain')
    elif 'Non-suicidal' in labelList:
        docLabel = avilableLabels.index('Non-suicidal')
    else:
        docLabel = avilableLabels.index('noLabel')
    return docLabel


def evaluationDocLevel(avilableLabels, prediction, y, docId, onehot=True):
    docLevelLableList = {}
    print(avilableLabels)
    correct = 0
    incorrect = 0
    evaluationListRecall=[[] for _ in range(len(avilableLabels))]
    evaluationListPrecision=[[] for _ in range(len(avilableLabels))]
    for j in range(len(evaluationListRecall)):
        evaluationListRecall[j] = [0.01,0.01] #[correct, incorrect]
        evaluationListPrecision[j] = [0.01,0.01]

    for i in range(len(prediction)):
        tmpPred = []
        tmpGold = []
        currentLabel = docId[i]
        for j in range(len(prediction[i])):
            currentPrediction = prediction[i][j]
            currentGold = y[i][j]
            if len(avilableLabels) > 2:
                if onehot == True:
                    yidx = get4classID(currentGold)
                    pidx = get4classID(currentPrediction)
                else:
                    pidx = currentPrediction
                    yidx = currentGold
            else:
                if onehot == True:
                    yidx = get2classID(currentGold)
                    pidx = get2classID(currentPrediction)
                else:
                    pidx = currentPrediction
                    yidx = currentGold
            tmpPred.append(avilableLabels[pidx])
            tmpGold.append(avilableLabels[yidx])


        plabel = transSent2DocLabel(avilableLabels, tmpPred)
        ylabel = transSent2DocLabel(avilableLabels, tmpGold)
        docLevelLableList[currentLabel] = [plabel, ylabel]
        if plabel == ylabel:
            correct += 1
            evaluationListRecall[plabel][0] += 1
            evaluationListPrecision[plabel][0] += 1
        else:
            incorrect += 1
            evaluationListPrecision[plabel][1] += 1
            evaluationListRecall[ylabel][1] += 1
    print(evaluationListRecall, evaluationListPrecision)
    accuracy = correct/(correct+incorrect)
    print('accuracy ', accuracy)
    for i in range(len(avilableLabels)):
        print(avilableLabels[i])
        recall =  evaluationListRecall[i][0]/ (evaluationListRecall[i][0]+evaluationListRecall[i][1])
        precision = evaluationListPrecision[i][0] / (evaluationListPrecision[i][0]+ evaluationListPrecision[i][1])
        fmeasure = 2*(recall * precision)/(recall+precision)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f-measure ', fmeasure)
    return docLevelLableList


def evaluationPatLevel(avilableLabels, patientDict, evaluationDoc):
    print(avilableLabels)
    correct = 0
    incorrect = 0
    evaluationListRecall=[[] for _ in range(len(avilableLabels))]
    evaluationListPrecision=[[] for _ in range(len(avilableLabels))]
    for j in range(len(evaluationListRecall)):
        evaluationListRecall[j] = [0.01,0.01] #[correct, incorrect]
        evaluationListPrecision[j] = [0.01,0.01]

    for eachPatient in patientDict:
        tmpPred = []
        tmpGold = []
        for documentId in patientDict[eachPatient]:
            pidx = evaluationDoc[documentId][0] 
            yidx = evaluationDoc[documentId][1]   
            tmpPred.append(avilableLabels[pidx])
            tmpGold.append(avilableLabels[yidx])
        plabel = transSent2DocLabel(avilableLabels, tmpPred)
        ylabel = transSent2DocLabel(avilableLabels, tmpGold)
        if plabel == ylabel:
            correct += 1
            evaluationListRecall[plabel][0] += 1
            evaluationListPrecision[plabel][0] += 1
        else:
            incorrect += 1
            evaluationListPrecision[plabel][1] += 1
            evaluationListRecall[ylabel][1] += 1
    print(evaluationListRecall, evaluationListPrecision)
    accuracy = correct/(correct+incorrect)
    print('accuracy ', accuracy)
    for i in range(len(avilableLabels)):
        print(avilableLabels[i])
        recall =  evaluationListRecall[i][0]/ (evaluationListRecall[i][0]+evaluationListRecall[i][1])
        precision = evaluationListPrecision[i][0] / (evaluationListPrecision[i][0]+ evaluationListPrecision[i][1])
        fmeasure = 2*(recall * precision)/(recall+precision)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f-measure ', fmeasure)



def bow_training_sents(labeledList, idlist):
    x=[]
    for i in idlist:
        doc = labeledList[i]
        for line in doc[2]:
                x += (line[1].lower()).split()
        print(i)
    return x

def svm_sents(labeledList, idlist):
    x=[]
    for i in idlist:
        doc = labeledList[i]
        for line in doc[2]:
                x += (line[1].lower())
    return x

