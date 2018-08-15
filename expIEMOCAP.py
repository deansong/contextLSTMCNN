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
from w2vFOFE import FOFEW2V, DataProcess, DataProcess2class
#from genLabel import genSentLabel
import gc
from optparse import OptionParser
from keras.models import load_model
import time
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from keras import regularizers
from keras.callbacks import EarlyStopping
from  BuildModels import Build_NN_Model


class IEMOCAPdata(DataProcess):
    def __init__(self):
        DataProcess.__init__(self)
        #self.avilableLabels=['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'fea', 'sur', 'dis', 'oth','xxx']
        #self.avilableLabels=['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'oth', 'xxx']
        #self.avilableLabels=['ang', 'sad', 'neu', 'exc', 'oth','xxx']
        #self.usedLabels = ['ang', 'sad', 'neu', 'exc']
        self.avilableLabels=['ang', 'sad', 'neu', 'exc']


    def doc2W2Vdoc(self, dataList, idlist):
        x=[]
        y=[]
        all_sents=[]
        total = [0] * len(self.avilableLabels)
        for eID in idlist:
            doc = dataList[eID][1]
            currentDocListx = []
            currentDocListy = []
            for line in doc:
                #print(line)
                currentLabel = line[3]
                segLabel = line[0]
                segTok = segLabel.split('_')
                #print(segLabel, segTok[-1][0])
                speaker = segTok[-1][0]
                if currentLabel in self.avilableLabels:
                    idx = self.avilableLabels.index(currentLabel)
                    total[idx] +=1
                w2vList,fofeCode = self.sentW2v(line[2], self.ebd_size)
                currentDocListx.append([w2vList, fofeCode, speaker])
                currentDocListy.append(currentLabel)
            y.append(currentDocListy)
            x.append(currentDocListx)
        return x,y,total

    def w2vdoc2fofesent(self, x,y, windowSize = 99999, alpha = 0.1):
        newx=[]
        newxpre=[]
        newxlat=[]
        newy=[]
        for i in range(len(x)):
            for j in range(len(x[i])):
                currentLabel = y[i][j]
                currentx= x[i][j][0]
                currentSpeak = x[i][j][2]
                if currentLabel in self.avilableLabels:
                    if j == 0:
                        prex = [0]*self.ebd_size
                        latx = [0]*self.ebd_size
                    else:
                        preHead = j - windowSize
                        if preHead > 0:
                            prex = self.sent_fofe_window(x[i][preHead:j], currentSpeak, alpha = alpha)
                            latx = self.sent_fofe_window(x[i][preHead:j], currentSpeak, reverse = True, alpha = alpha)
                        else:
                            prex = self.sent_fofe_window(x[i][0:j], currentSpeak, alpha = alpha)
                            latx = self.sent_fofe_window(x[i][0:j], currentSpeak, reverse = True, alpha = alpha)
                    newx.append(currentx)
                    newxpre.append(prex)
                    newxlat.append(latx)
                    newy.append(label_binarize([currentLabel], self.avilableLabels).tolist()[0])
        return np.array(newx), np.array(newxpre), np.array(newxlat), np.array(newy)

    def w2vdoc2w2vsent(self, x,y, windowSize = 99999, alpha = 0.1):
        newx=[]
        newxpre=[]
        newxlat=[]
        newy=[]
        for i in range(len(x)):
            for j in range(len(x[i])):
                currentLabel = y[i][j]
                currentx= x[i][j][0]
                currentSpeak = x[i][j][2]
                if currentLabel in self.avilableLabels:
                    if j == 0:
                        prex = [[0]*self.ebd_size]*self.maxSentLen
                        latx = [[0]*self.ebd_size]*self.maxSentLen
                    else:
                        preHead = j - windowSize
                        if preHead > 0:
                            prex = self.sent_w2v_window(x[i][preHead:j], currentSpeak, alpha = alpha)
                            latx = self.sent_w2v_window(x[i][preHead:j], currentSpeak, reverse = True, alpha = alpha)
                        else:
                            prex = self.sent_w2v_window(x[i][0:j], currentSpeak, alpha = alpha)
                            latx = self.sent_w2v_window(x[i][0:j], currentSpeak, reverse = True, alpha = alpha)
                    newx.append(currentx)
                    newxpre.append(prex)
                    newxlat.append(latx)
                    newy.append(label_binarize([currentLabel], self.avilableLabels).tolist()[0])
        return np.array(newx), np.array(newxpre), np.array(newxlat), np.array(newy)

    def sent_w2v_window(self, doc_w2v_list, currentSpeak, reverse = False, alpha = 0.1):
        doc_w2v_list.reverse() # Only output previous 1, so reverse
        start = 0
        fofeCode = [[0]*self.ebd_size]*self.maxSentLen
        for sent_list in doc_w2v_list:
            currentfofe = sent_list[0]
            speaker = sent_list[2]
            if (currentSpeak == speaker and reverse == False) or (currentSpeak != speaker and reverse == True):
                fofeCode = currentfofe
                break
        return fofeCode


    def sent_fofe_window(self, doc_w2v_list, currentSpeak, reverse = False, alpha = 0.1):
        start = 0
        fofeCode = np.array([0]*self.ebd_size)
        for sent_list in doc_w2v_list:
            currentfofe = sent_list[1]
            speaker = sent_list[2]
            if (currentSpeak == speaker and reverse == False) or (currentSpeak != speaker and reverse == True):
                if start == 0:
                    fofeCode = currentfofe
                    start = 1
                else:
                    try:
                        fofeCode = fofeCode*alpha + currentfofe
                    except:
                        print(currentfofe, fofeCode)
        #print(len(fofeCode))
        return fofeCode.tolist()

def loadTranscription(transcriptionDir, evaluationDir = None, avilableLabels=None):
    ps = PorterStemmer()
    translist=[]
    allSentence = []
    evaluationDict = {}
    for txtFile in glob.glob(transcriptionDir+'/*.txt'):
        fileNameToks = txtFile.split('/')
        fullFileName = fileNameToks[-1]
        translist.append([fullFileName, []])
        with open(txtFile,'r') as ftxt:
            doc = ftxt.readlines()
        if evaluationDir:
            evaluationFile = evaluationDir+fullFileName
            with open(evaluationFile,'r') as feva:
                evaluation = feva.readlines()
            for line in evaluation:
                m=re.match('\[.*\]',line)
                if m:
                    lineTok = line.split('\t')
                    segLabel = lineTok[1]
                    emo = lineTok[2]
                    evaluationDict[segLabel] = emo
        for line in doc:
            m = re.match('Ses.*\[.*\]\: .*', line)
            if m:
                lineTok = line.split(' ')
                #print lineTok
                segLabel = lineTok[0]
                duration = lineTok[1]
                rawtext = ' '.join(lineTok[2:])
                if segLabel in evaluationDict:
                    emotion = evaluationDict[segLabel]
                else:
                    emotion = None
                text = rawtext.strip()
                textTok = word_tokenize(text)
                textTokStem = [ps.stem(word) for word in textTok]
                if avilableLabels:
                    if emotion in avilableLabels:
                        translist[-1][1].append([segLabel, duration, textTokStem, emotion])
                    else:
                        translist[-1][1].append([segLabel, duration, textTokStem, 'oth'])
                else:
                    translist[-1][1].append([segLabel, duration, textTokStem, emotion]) 
    return translist

def w2vSentenceExtract(dataList, trainIDs):
    rawSentences = []
    for eID in trainIDs:
        currentTXT = dataList[eID][1]
        for item in currentTXT:
            #print(item)
            sentence = item[2]
            rawSentences.append(sentence)
    return rawSentences
        


parser = OptionParser()
parser.add_option("--transcriptions", dest="transcriptionDir", help="transcription directory")
parser.add_option("--evaluation", dest="evaluationDir", help="evaluation directory")
parser.add_option("--train", dest="train", help="train model")
parser.add_option("--saveData", dest="saveData", action='store_true',help="train model")
parser.add_option("--test", dest="test", action='store_true',help="test model")
parser.add_option("--trainTestSplit", dest="trainTestSplit",action='store_true', help = "split file into training (80%) and testing parts(20%)")
parser.add_option("--saveTrainTest", dest="savetraintest", help = "save train test split")
parser.add_option("--nFoldsSplit", dest="nFoldsSplit", help = "prefix of n folds split")
parser.add_option("--saveModel", dest="saveModel", help = "save trained model")
parser.add_option("--epochs", dest="epochs", default=50, help = "number of epochs for training, default 50", type="int")
parser.add_option("--testIDs", dest="testIDs", help="load test IDs")
parser.add_option("--trainIDs", dest="trainIDs", help="load train IDs")
parser.add_option("--windowSize", dest="windowsSize", default=999999, help = "window size", type="int")
parser.add_option("--loadModel", dest="loadModel", help="load trianed model")
parser.add_option("--sentAlpha", dest="sentAlpha", default=0.1, help = "sent alpha", type="float")
parser.add_option("--wordAlpha", dest="wordAlpha", default=0.95, help = "word alpha", type="float")
parser.add_option("--experiment", dest="experiment", default=False ,action='store_true',help="compare with other models")
parser.add_option("--blstmBlstmcnn", dest="blstmBlstmcnn",default=False ,action='store_true',help="train blstmBlstmcnn")
parser.add_option("--loadW2V", dest="loadW2V", help="load trianed w2v")


blstmcnnFofe = None
blstmcnn = None
blstmOnly = None
cnnOnly = None
blstmBlstmcnn = None
svmModel = None
maxEntModel = None



w2v = IEMOCAPdata()


options, arguments = parser.parse_args()
w2v.wordAlpha = options.wordAlpha
print(options.epochs)
print(options.windowsSize)
if options.transcriptionDir and options.evaluationDir:
    dataList = loadTranscription(options.transcriptionDir, options.evaluationDir, w2v.avilableLabels)
if options.loadW2V:
    w2v.load_w2v(options.loadW2V)

if options.loadModel:
    w2v.load_w2v(options.loadModel+'w2v.model')
    if options.blstmBlstmcnn:
        blstmBlstmcnn = load_model(options.loadModel+'blstmBlstmcnn.model')
    else:
        blstmcnnFofe = load_model(options.loadModel+'blstmcnnFofe.model')
    if options.experiment:
        blstmcnn = load_model(options.loadModel+'blstmcnn.model')
        blstmOnly = load_model(options.loadModel+'blstm.model')
        cnnOnly = load_model(options.loadModel+'cnn.model')

allSentIds = range(len(dataList))
trainIDs = allSentIds
testIDs = allSentIds

if options.nFoldsSplit:
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


if options.trainIDs:
    with open(options.trainIDs,'rb') as fin:
        trainIDs = pickle.load(fin)

if options.testIDs:
    with open(options.testIDs,'rb') as fin:
        testIDs = pickle.load(fin)

if options.trainTestSplit:
    trainIDs, testIDs = train_test_split(allSentIds, test_size=0.2)


if options.savetraintest:
    with open(options.savetraintest+'trainid.pkl', 'wb') as fpkl:
        pickle.dump(trainIDs, fpkl)
    with open(options.savetraintest+'testid.pkl', 'wb') as fpkl:
        pickle.dump(testIDs, fpkl)

if options.train or options.saveModel:
    if options.loadW2V:
        print('load w2v')
    else:
        print('train w2v')
        w2v.sentence = w2vSentenceExtract(dataList, trainIDs)
        w2v.build_w2v()
        w2v.save_w2v(options.train+'w2v.model')
    xall, yall,total = w2v.doc2W2Vdoc(dataList, trainIDs)
    print(total)
    xtrain, pretrain, lattrain, ytrain= w2v.w2vdoc2fofesent(xall,yall, windowSize=options.windowsSize, alpha=options.sentAlpha)

    maxTrain = max(total)
    maxTrainIdx = total.index(maxTrain)
    c_weight = {}
    for tcIndex in range(len(total)):
        c_weight[tcIndex] = maxTrain / total[tcIndex]
    print(c_weight)

    nn_model = Build_NN_Model(len(w2v.avilableLabels))
    if options.blstmBlstmcnn:
        xtrain, pretrain, lattrain, ytrain= w2v.w2vdoc2w2vsent(xall,yall)
        blstmBlstmcnn = nn_model.blstmBlstmcnn()
        trainLoss = blstmBlstmcnn.fit([np.array(xtrain),np.array(pretrain),np.array(lattrain)], np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        blstmBlstmcnn.save(options.train+'blstmBlstmcnn.model')
    else:

        blstmcnnFofe = nn_model.contextBlstmcnn()
        trainLoss = blstmcnnFofe.fit([np.array(xtrain),np.array(pretrain),np.array(lattrain)], np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        blstmcnnFofe.save(options.train+'blstmcnnFofe.model')
    if options.experiment:
        blstmcnn = nn_model.blstmcnn()
        blstmOnly = nn_model.blstm_only()
        cnnOnly = nn_model.cnn_only()

        trainLoss = blstmcnn.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        trainLoss = blstmOnly.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)
        trainLoss = cnnOnly.fit(np.array(xtrain), np.array(ytrain), epochs=options.epochs, batch_size=64, class_weight=c_weight)

        blstmcnn.save(options.train+'blstmcnn.model')
        blstmOnly.save(options.train+'blstm.model')
        cnnOnly.save(options.train+'cnn.model')


if options.test:
    xall, yall,total = w2v.doc2W2Vdoc(dataList, testIDs)
    xtest, pretest, lattest, ytest = w2v.w2vdoc2fofesent(xall,yall, windowSize=options.windowsSize, alpha=options.sentAlpha)
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
        xtest, pretest, lattest, ytest= w2v.w2vdoc2w2vsent(xall,yall)
        blstmBlstmcnn.summary()
        blstmBlstmcnnPrediction = blstmBlstmcnn.predict([np.array(xtest), np.array(pretest), np.array(lattest)])
        evaluation(w2v.avilableLabels, blstmBlstmcnnPrediction ,ytest)
