from __future__ import division
import sys
import os
import glob
import pickle
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer
import socket
import json


#from keras.preprocessing.text import Tokenizer

#global word_punct_tokenizer
#global kerasTokenizer

global HOST 
global PORT 
HOST = "localhost"
PORT = 7788

def recvDocFromJava(sock):
    eod = False
    docIDList = []
    while(eod ==False):
        data_recv = sock.recv(1024)
        data = json.loads(data_recv)
        eod = data['eod']
        startid = data['start']
        endid = data['end']
        sock.sendall("success\n")
        #print(startid, endid)
        if eod == False:
            docIDList.append((startid, endid))
    return docIDList


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def nolabelappend(tmpNolabelSentsTok, word_punct_tokenizer):
    tmpList = []
    for tmpNolabelSent in tmpNolabelSentsTok:
        tmpNolabelSentSplit = sent_tokenize(tmpNolabelSent)
        for tmpNolabelSentSplitSent in tmpNolabelSentSplit:
            tmpList.append(['noLabel', ' '.join(word_punct_tokenizer.tokenize(tmpNolabelSentSplitSent))])
    return tmpList


def getAnnoList(annotations):
    annoList = []
    for annotation in annotations:
        annoTOK = annotation.split('\t')
        annoId = annoTOK[0]
        annoSent = annoTOK[2]
        #print(annoId, annoSent)
        if annoId[0] == 'T':
            annoTOK1 = annoTOK[1].split(' ')
            annoLabel = annoTOK1[0]
            #print(annoLabel)
            annoLabelStart = int(annoTOK1[1])
            annoLabelEnd = int(annoTOK1[-1])
            annoList.append([annoId,annoLabel,annoLabelStart,annoLabelEnd,annoSent])
    sorted_annoList = sorted(annoList, key=lambda annox: annox[2])
    return sorted_annoList

def getSentList(docSent):
    #print(docSent)
    sentList = []
    i = 0
    idend = 0
    idstart = 0
    lineSplited = docSent.split('\n')
    for item in lineSplited:
        sents = sent_tokenize(item)
        for sent in sents:
            idend = idstart+len(sent)-1
            while (docSent[idend] == ' ' or docSent[idend] == '\n') and idend < len(docSent)-1:
                idend += 1
            sentList.append([sent, idstart, idend])
            idstart = idend+1
        
    return sentList



def splitAccoLabel_nosplit(annotations, doc, doclist, word_punct_tokenizer):
    #print(annotations)
    docSent = ''.join(doc)
    annoList = getAnnoList(annotations)
    #print(annoList)
    labelledSentList=[]
    idx=0
    #print('Annolist: ', annoList)
    sentid = 0
    diffCount = 0
    replacelabelCount = 0
    overSentlabelCount = 0
    for annotation in annoList:
        annoLabel = annotation[1]
        annoLabelStart = annotation[2]
        annoLabelEnd = annotation[3]
        #print(annoLabel, annoLabelStart, annoLabelEnd, annotation[4])
        hit = False
        sentPreBackup = ''
        while(sentid<len(doclist) and hit == False):
            currentSent = doclist[sentid]
            sentStart = currentSent[0]
            sentEnd = currentSent[1]
            sentText = docSent[sentStart:sentEnd]
            #print(sentText, sentStart, sentEnd)
            if sentStart <= annoLabelStart and sentEnd >= annoLabelEnd:
                hit = True
                sentLabel = annoLabel
                sentPreBackup = ''
                labelledSentList.append([sentLabel, ' '.join(word_punct_tokenizer.tokenize(sentText))])
                #print('normal sentence:', labelledSentList[-1])
                #print(labelledSentList[-1])
            elif sentEnd < annoLabelStart:
                sentLabel = 'noLabel'
                labelledSentList.append([sentLabel, ' '.join(word_punct_tokenizer.tokenize(sentText))])
                #print('nolabel sentence:', labelledSentList[-1])
            elif sentStart <= annoLabelStart and sentEnd > annoLabelStart:
                #print('save label')
                sentPreBackup+=sentText+' '
            elif sentEnd >= annoLabelEnd and sentStart >= annoLabelStart:
                #print(annoLabel, annoLabelStart, annoLabelEnd, annotation[4])
                #assert len(sentPreBackup) > 0
                if len(sentPreBackup) > 0:
                    hit = True
                    sentLabel = annoLabel
                    combinedSent = sentPreBackup+sentText
                    labelledSentList.append([sentLabel, ' '.join(word_punct_tokenizer.tokenize(combinedSent))])
                    sentPreBackup=''
                    overSentlabelCount += 1
                    #print('over sentence:', labelledSentList[-1])
                else:
                    hit = True
                    lastLabel = labelledSentList[-1][0]
                    sentLabel = annoLabel
                    #assert lastLabel != 'noLabel'
                    #assert sentLabel != 'noLabel'
                    if lastLabel != 'noLabel':
                        sentid-=1
                        if lastLabel == 'Uncertain' or sentLabel == 'Uncertain':
                            labelledSentList[-1][0] = 'Uncertain'
                        elif lastLabel == 'Suicidality' or sentLabel == 'Suicidality':
                            labelledSentList[-1][0] = 'Suicidality'
                        else:
                            labelledSentList[-1][0] = 'Non-suicidal'
                        replacelabelCount += 1
                        if lastLabel != sentLabel:
                            diffCount +=1
                            print('replace sentence:', labelledSentList[-1])
                    else:
                        lastSent = doclist[sentid-1]
                        lastEndId = lastSent[1]
                        iddiff = sentStart - lastEndId
                        assert iddiff > 1
                        labelledSentList.append([sentLabel, ' '.join(word_punct_tokenizer.tokenize(sentText))])
                        #print('normal sentence:', labelledSentList[-1])
            #print(labelledSentList)
            sentid+=1
        #print(labelledSentList[-1])
    while(sentid<len(doclist)):
        currentSent = doclist[sentid]
        sentStart = currentSent[0]
        sentEnd = currentSent[1]
        sentText = docSent[sentStart:sentEnd]
        sentLabel = 'noLabel'
        labelledSentList.append([sentLabel, ' '.join(word_punct_tokenizer.tokenize(sentText))])
        sentid+=1
    return labelledSentList, replacelabelCount, overSentlabelCount, diffCount


def genSentLabel(inputFolder):
    tr = 0
    to = 0
    td = 0
    topwords = 10000
    word_punct_tokenizer = WordPunctTokenizer()
    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.connect((HOST, PORT))
    count = 0
    allLabeledList = []
    for subFolder in glob.glob(inputFolder+'/*'):
        #print(subFolder)
        for txtFile in glob.glob(subFolder+'/*.txt'):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            #print(txtFile)
            with open(txtFile,'r') as fin:
                doc = fin.readlines()
            #kerasTokenizer.fit_on_texts(doc)
            fileNameToks = txtFile.split('/')
            #print(fileNameToks)
            subfolderPrefix = fileNameToks[-2]
            prefix=fileNameToks[-1].split('.')[0]
            #print(prefix)
            annoFile= prefix+'.ann'
            annoFilePath = subFolder+'/'+annoFile
            #print(annoFilePath)
            with open(annoFilePath,'r') as fin:
                annotations = fin.readlines()
            #print(doc)
            #print(annotations)
            toServer = {'fromClient':{'txt':txtFile, 'eod':True}}
            sock.sendall(json.dumps(toServer, cls=MyEncoder)+"\n")
            doclist = recvDocFromJava(sock)
            labelledSentList, replacelabelCount, overSentlabelCount, diffCount = splitAccoLabel_nosplit(annotations, doc, doclist, word_punct_tokenizer)
            tr+=replacelabelCount
            to+=overSentlabelCount
            td+=diffCount
            #labelledSentList = splitAccoLabel(annotations, doc)
            allLabeledList.append([subfolderPrefix,prefix,labelledSentList])
            sock.close()
        count +=1
        #if count > 3:
        #    break
    #print(allLabeledList)
    print(tr,to,td)
    return allLabeledList 

if __name__ == "__main__":
    inputFolder = sys.argv[1]
    outputPrefix = sys.argv[2]
    allLabeledList = genSentLabel(inputFolder)
    labeledFileName = outputPrefix+'_labeledSent.pkl'
    with open(labeledFileName,'wb') as fp:
        pickle.dump(allLabeledList,fp)
