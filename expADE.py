from __future__ import division
import sys
import os
import glob
import pickle
#import nltk.data
import numpy as np
from keras.preprocessing.text import Tokenizer
#from fofeModel import *
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
sys.path.append("ulti")
from ulti import *
from w2vFOFE import *
from optparse import OptionParser
from keras.models import load_model
import time
import re
from nltk.stem import PorterStemmer
from keras import regularizers
from keras.callbacks import EarlyStopping

from pubmed_lookup import PubMedLookup, Publication
from nltk.tokenize import WordPunctTokenizer
from  BuildModels import Build_NN_Model
from nltk.tokenize import sent_tokenize


class ADEdata(DataProcess2class):
    def __init__(self):
        DataProcess2class.__init__(self)
        self.avilableLabels=['neg','pos']

    def w2v_training_sents(self, dataList, trainID):
        word_punct_tokenizer = WordPunctTokenizer()
        x=[]
        for currentId in trainID:
            currentData = dataList[currentId]
            currentSent = currentData[2]
            currentPreList = currentData[3]
            currentLatList = currentData[4]
            x.append(' '.join(word_punct_tokenizer.tokenize(currentSent)))
            for item in currentPreList:
                x.append(' '.join(word_punct_tokenizer.tokenize(item)))
            for item in currentLatList:
                x.append(' '.join(word_punct_tokenizer.tokenize(item)))
        return x 


    def fofeEncoding(self, sentList, word_punct_tokenizer, sentLevelAlpha=0.1, reverse=False):
        fofeList = []
        for currentSent in sentList:
            w2vList, fofeCode = self.sentW2v(word_punct_tokenizer.tokenize(currentSent), self.ebd_size)
            fofeList.append([w2vList, fofeCode])
        docLeveFofe = self.sent_fofe_window(fofeList, reverse=reverse, alpha=sentLevelAlpha)
        return docLeveFofe

    def w2vEncoding(self, sentList, word_punct_tokenizer, sentLevelAlpha=0.1, reverse=False):
        fofeList = []
        for currentSent in sentList:
            w2vList, fofeCode = self.sentW2v(word_punct_tokenizer.tokenize(currentSent), self.ebd_size)
            fofeList.append([w2vList, fofeCode])
        docLeveFofe = self.sent_w2v_window(fofeList, reverse=reverse, alpha=sentLevelAlpha)
        return docLeveFofe

    def sent_w2v_window(self, doc_w2v_list, reverse = False, alpha = 0.1):
        if reverse == True:
            doc_w2v_list.reverse()
        for sent_list in doc_w2v_list:
            currentfofe = sent_list[0]
            fofeCode = currentfofe
            break
        return fofeCode


    def transferDataw2v(self, allLabeledList, trainID, alpha=0.1):
        word_punct_tokenizer = WordPunctTokenizer()
        total = [0] * len(self.avilableLabels)
        xtrain = []
        pretrain = []
        lattrain = []
        ytrain = []
        for currentId in trainID:
            currentData = dataList[currentId]
            currentLabel = currentData[1]
            currentSent = currentData[2]
            currentPreList = currentData[3]
            currentLatList = currentData[4]
            if currentLabel in self.avilableLabels:
                idx = self.avilableLabels.index(currentLabel)
                total[idx] +=1
                binLabel = label_binarize([currentLabel], self.avilableLabels).tolist()[0]
                w2vList,fofeCode = self.sentW2v(word_punct_tokenizer.tokenize(currentSent), self.ebd_size)
                xtrain.append(w2vList)
                prefofe = self.w2vEncoding(currentPreList, word_punct_tokenizer, sentLevelAlpha=alpha)
                latfofe = self.w2vEncoding(currentPreList, word_punct_tokenizer, reverse=True, sentLevelAlpha=alpha)
                pretrain.append(prefofe)
                lattrain.append(latfofe)
                ytrain.append(binLabel)
        return xtrain, pretrain, lattrain, ytrain, total


         
    def transferData(self, allLabeledList, trainID, alpha=0.1):
        word_punct_tokenizer = WordPunctTokenizer()
        total = [0] * len(self.avilableLabels)
        xtrain = []
        pretrain = []
        lattrain = [] 
        ytrain = [] 
        for currentId in trainID:
            currentData = dataList[currentId]
            currentLabel = currentData[1]
            currentSent = currentData[2]
            currentPreList = currentData[3]
            currentLatList = currentData[4]
            if currentLabel in self.avilableLabels:
                idx = self.avilableLabels.index(currentLabel)
                total[idx] +=1
                binLabel = label_binarize([currentLabel], self.avilableLabels).tolist()[0]
                w2vList,fofeCode = self.sentW2v(word_punct_tokenizer.tokenize(currentSent), self.ebd_size)
                xtrain.append(w2vList)
                prefofe = self.fofeEncoding(currentPreList, word_punct_tokenizer, sentLevelAlpha=alpha)
                latfofe = self.fofeEncoding(currentPreList, word_punct_tokenizer, reverse=True, sentLevelAlpha=alpha)
                pretrain.append(prefofe)
                lattrain.append(latfofe)
                ytrain.append(binLabel)
        return xtrain, pretrain, lattrain, ytrain, total

def readNegLine(negLine):
    lineTok = negLine.split(' ')
    docID = lineTok[0]
    label = 'neg'
    sentence = ' '.join(lineTok[2:]).strip()
    return [sentence,docID,label]


def readPosLine(posLine):
    lineTok = posLine.split('|')
    docID = lineTok[0]
    sentence = lineTok[1]
    label = 'pos'
    return [sentence,docID,label]


def loadADEcorpus(negFile, posFile, avilableLabels=None):
    email = ''
    urlPre = 'http://www.ncbi.nlm.nih.gov/pubmed/'

    allAbstract={}
    currentSentList = []
    sentPreList = []
    sentLatList = []
    labelList = []
    i=0
    allFiles = [negFile, posFile]
    for fileId in range(len(allFiles)):
        with open(allFiles[fileId],'r') as fin:
             for line in fin:
                 if fileId == 0:
                     sentence, docID, label = readNegLine(line)
                 else:
                     sentence, docID, label = readPosLine(line)
                 try:
                     if docID not in allAbstract:
                         fullUrl = urlPre+docID
                         lookup = PubMedLookup(fullUrl, email)
                         publication = Publication(lookup)
                         abstract = publication.abstract
                         allAbstract[docID] = abstract
                     else:
                         abstract = allAbstract[docID]

                     abstractSplit = abstract.split(sentence)
                     if len(abstractSplit) == 2:
                             print(sentence)
                             currentSentList.append(sentence)
                             print(abstractSplit[0])
                             sentPreList.append(abstractSplit[0])
                             print(abstractSplit[1])
                             sentLatList.append(abstractSplit[1])
                             labelList.append(label)
                 except:
                     print('cant find file')
                 i+=1
                 print(i, len(currentSentList))
    return [currentSentList,sentPreList,sentPreList, labelList, allAbstract]
             

def w2vSentenceExtract(dataList, idList):
    rawSentences = []
    usedId = []
    word_punct_tokenizer = WordPunctTokenizer()
    for listId in idList:
        docId = dataList[listId][0]
        if docId not in usedId:
            usedId.append(docId)
            presentList = dataList[listId][3]
            currentSent = dataList[listId][2]
            latsentList = dataList[listId][4]
            sentList = presentList + latsentList
            sentList.append(currentSent)
            for sentence in sentList:
                textTok = word_punct_tokenizer.tokenize(sentence)
                rawSentences.append(textTok)
    return rawSentences
        

parser = OptionParser()
parser.add_option("--txtFile", dest="txtFile", help="txt input file")
parser.add_option("--pklFile", dest="pklFile", help="pickled input file")
parser.add_option("--train", dest="train", help="train model")
parser.add_option("--test", dest="test", action='store_true',help="test model")
parser.add_option("--trainTestSplit", dest="trainTestSplit",action='store_true', help = "split file into training (80%) and testing parts(20%)")
parser.add_option("--saveTrainTest", dest="savetraintest", help = "save train test split")
parser.add_option("--saveModel", dest="saveModel", help = "save trained model")
parser.add_option("--epochs", dest="epochs", default=50, help = "number of epochs for training, default 50", type="int")
parser.add_option("--testIDs", dest="testIDs", help="load test IDs")
parser.add_option("--trainIDs", dest="trainIDs", help="load train IDs")
parser.add_option("--windowSize", dest="windowsSize", default=1, help = "window size", type="int")
parser.add_option("--loadModel", dest="loadModel", help="load trianed model")
parser.add_option("--nFoldsSplit", dest="nFoldsSplit", help = "prefix of n folds split")
parser.add_option("--sentenceLevelEvaluation", dest="sentenceLevelEvaluation",action='store_true', help = "sentence level evaluation")
parser.add_option("--experiment", dest="experiment", default=False ,action='store_true',help="compare with other models")
parser.add_option("--sentAlpha", dest="sentAlpha", default=0.1, help = "sent alpha", type="float")
parser.add_option("--blstmBlstmcnn", dest="blstmBlstmcnn",default=False ,action='store_true',help="train blstmBlstmcnn")
parser.add_option("--loadW2V", dest="loadW2V", help="load trianed w2v")
parser.add_option("--wordAlpha", dest="wordAlpha", default=0.95, help = "word alpha", type="float")


w2v = ADEdata()
blstmcnnFofe = None
blstmcnn = None
blstmOnly = None
cnnOnly = None
blstmBlstmcnn = None
svmModel = None
maxEntModel = None


options, arguments = parser.parse_args()
w2v.wordAlpha = options.wordAlpha
print(options.epochs)
print(options.windowsSize)
if options.txtFile:
    print('not supported yet')
    #pass    

if options.pklFile:
    with open(options.pklFile,'rb') as fin:
        dataList = pickle.load(fin)

if options.nFoldsSplit:
    allSentIds = range(len(dataList))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)
    foldCount = 1
    for item in kf.split(allSentIds):
        currentTrainId = item[0].tolist()
        currentTestId = item[1].tolist()
        with open(options.nFoldsSplit+'trainIds'+str(foldCount)+'.pkl','wb') as fpkl:
            pickle.dump(currentTrainId, fpkl)
        with open(options.nFoldsSplit+'testIds'+str(foldCount)+'.pkl','wb') as fpkl:
            pickle.dump(currentTestId, fpkl)
        foldCount+=1

trainID = range(len(dataList))
testID = range(len(dataList))



if options.trainIDs:
    with open(options.trainIDs,'rb') as fin:
        trainID = pickle.load(fin)

if options.testIDs:
    with open(options.testIDs,'rb') as fin:
        testID = pickle.load(fin)

if options.loadW2V:
    w2v.load_w2v(options.loadW2V)


if options.loadModel:    
    w2v.load_w2v(options.loadModel+'w2v.model')
    if options.blstmBlstmcnn:
        blstmBlstmcnn = load_model(options.loadModel+'blstmBlstmcnn.model')
    elif options.experiment:
        blstmcnn = load_model(options.loadModel+'blstmcnn.model')
        blstmOnly = load_model(options.loadModel+'blstm.model')
        cnnOnly = load_model(options.loadModel+'cnn.model')
    else:
        blstmcnnFofe = load_model(options.loadModel+'blstmcnnFofe.model')


if options.train:
    if options.loadW2V:
        print('load w2v')
    else:
        print('train w2v')
        w2v.sentence = w2vSentenceExtract(dataList, trainID)
        w2v.build_w2v()
        w2v.save_w2v(options.train+'w2v.model')

    xtrain, pretrain, lattrain, ytrain, total = w2v.transferData(dataList, trainID, alpha=options.sentAlpha) 
    print(xtrain[0])
    print(pretrain[0])
    print(lattrain[0])
    print(ytrain[0]) 
    topClass = total.index(max(total))
    c_weight = {}
    for count_id in range(len(total)):
        if total[count_id] != 0:
            c_weight[count_id] = float(total[topClass]/total[count_id])
        else:
            c_weight[count_id] = 1.0
    print(c_weight)

    nn_model = Build_NN_Model(len(w2v.avilableLabels))
    if options.blstmBlstmcnn:
        xtrain, pretrain, lattrain, ytrain, total = w2v.transferDataw2v(dataList, trainID, alpha=options.sentAlpha)
        blstmBlstmcnn = nn_model.blstmBlstmcnn()
        startTime = time.time()
        trainLoss = blstmBlstmcnn.fit([np.array(xtrain),np.array(pretrain),np.array(lattrain)], np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        endTime = time.time()
        print(endTime - startTime)
        blstmBlstmcnn.save(options.train+'blstmBlstmcnn.model')

    elif options.experiment:
        blstmcnn = nn_model.blstmcnn()
        blstmOnly = nn_model.blstm_only()
        cnnOnly = nn_model.cnn_only()
        startTime = time.time()
        trainLoss = blstmcnn.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        endTime = time.time()
        print('blstmcnn training time')
        print(endTime - startTime)
        startTime = time.time()
        trainLoss = blstmOnly.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        endTime = time.time()
        print('blstmc training time')
        print(endTime - startTime)
        startTime = time.time()
        trainLoss = cnnOnly.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        endTime = time.time()
        print('cnn training time')
        print(endTime - startTime)
        blstmcnn.save(options.train+'blstmcnn.model')
        blstmOnly.save(options.train+'blstm.model')
        cnnOnly.save(options.train+'cnn.model')

    else:
        blstmcnnFofe = nn_model.contextBlstmcnn()
        startTime = time.time()
        trainLoss = blstmcnnFofe.fit([np.array(xtrain),np.array(pretrain),np.array(lattrain)], np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        endTime = time.time()
        print(endTime - startTime)
        blstmcnnFofe.save(options.train+'blstmcnnFofe.model')


if options.sentenceLevelEvaluation:
    xtest, pretest, lattest, ytest, total = w2v.transferData(dataList, testID, alpha=options.sentAlpha)
    xtest = np.array(xtest)
    pretest = np.array(pretest)
    lattest = np.array(lattest)
    ytest = np.array(ytest)

    if blstmcnnFofe:
        print('blstmcnnFofe Evaluation')
        blstmcnnfofeprediction = blstmcnnFofe.predict([xtest,pretest,lattest])
        evaluation(w2v.avilableLabels, blstmcnnfofeprediction ,ytest)

    if blstmcnn:
        print('blstmcnn Evaluation')
        blstmcnnprediction = blstmcnn.predict(xtest)
        evaluation(w2v.avilableLabels, blstmcnnprediction, ytest)

    if blstmOnly:
        print('blstm Evaluation')
        blstmOnlyprediction = blstmOnly.predict(xtest)
        evaluation(w2v.avilableLabels, blstmOnlyprediction, ytest)

    if cnnOnly:
        print('cnn Evaluation')
        cnnOnlyprediction = cnnOnly.predict(xtest)
        evaluation(w2v.avilableLabels, cnnOnlyprediction, ytest)

    if blstmBlstmcnn:
        print('blstmcnnblstm Evaluation')
        xtest, pretest, lattest, ytest, total = w2v.transferDataw2v(dataList, testID, alpha=options.sentAlpha)
        blstmBlstmcnn.summary()
        blstmBlstmcnnPrediction = blstmBlstmcnn.predict([np.array(xtest), np.array(pretest), np.array(lattest)])
        evaluation(w2v.avilableLabels, blstmBlstmcnnPrediction ,ytest)

