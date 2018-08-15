# 1829 neg 1680 pos filted out

import pickle
import sys
import json
import socket
from io import BytesIO
import time

neginputFile = sys.argv[1]
posinputFile = sys.argv[2]
abstractPklFile = sys.argv[3]
outputFile = sys.argv[4]

global HOST
global PORT
HOST = "localhost"
PORT = 7789

def splitSent(doc, splitIdList):
    splitedDoc = []
    for splitId in splitIdList:
        splitedDoc.append(doc[splitId[0]:splitId[1]])
    return splitedDoc


def recvDocFromJava(sock):
    eod = False
    docIDList = []
    while(eod ==False):
        data_recv = sock.recv(1024)
        data = json.loads(data_recv)
        eod = data['eod']
        startid = data['start']
        endid = data['end']
        sock.sendall(b"success\n")
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

def sendDoc2Java(doc, maxChar=2048):
    for _ in range(5):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            break
        except:
            print('Connection Failed! Retry connection after 5 seconds... ')
            sock.close()
            time.sleep(5)
            #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #sock.connect((HOST, PORT))

    previ = 0
    i = maxChar
    eod = False
    docLen = len(doc)

    while(eod == False):
        subdoc = doc[previ:i]
        if i < len(doc):
            eod = False
        else:
            eod = True
        #print(subdoc)
        testSend = {'fromClient':{'doc':subdoc, 'eod':eod}}
        previ = i
        i += maxChar
        dumpedJson=json.dumps(testSend,cls=MyEncoder)
        print(dumpedJson)
        bytes_text = bytes(dumpedJson, 'utf-8')
        sock.sendall(bytes_text+b"\n")
    cleanedDoc = recvDocFromJava(sock)
    sock.close()
    return cleanedDoc



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

def gateSentSplit(preText, latText):
    splitIdListPre = sendDoc2Java(preText)
    splitedPre = splitSent(preText, splitIdListPre)
    splitIdListLat = sendDoc2Java(latText)
    splitedLat = splitSent(latText, splitIdListLat)
    return splitedPre, splitedLat 


with open(abstractPklFile,'rb') as fp:
    allAbstract = pickle.load(fp)

negNodoc = 0
posNodoc = 0
dataList = []
with open(outputFile,'w') as fo:
    with open(neginputFile,'r') as fneg:
        for line in fneg:
            sentence, docID, label = readNegLine(line)
            if docID in allAbstract:
                abstract = allAbstract[docID]
                abstractSplit = abstract.split(sentence)
                if len(abstractSplit) > 1:
                    splitedPre, splitedLat = gateSentSplit(abstractSplit[0].strip(), abstractSplit[1].strip())
                    fo.write(docID+'\t'+label+'\t'+sentence.strip()+'\t'+' '.join(splitedPre)+'\t'+' '.join(splitedLat)+'\n')
                    dataList.append([docID,label,sentence.strip(),splitedPre,splitedLat])
                else:
                    negNodoc+=1
            else:
                negNodoc+=1

    with open(posinputFile,'r') as fpos:
        for line in fpos:
            sentence, docID, label = readPosLine(line)
            if docID in allAbstract:
                abstract = allAbstract[docID]
                abstractSplit = abstract.split(sentence)
                if len(abstractSplit) > 1:
                    splitedPre, splitedLat = gateSentSplit(abstractSplit[0].strip(), abstractSplit[1].strip())
                    fo.write(docID+'\t'+label+'\t'+sentence.strip()+'\t'+' '.join(splitedPre)+'\t'+' '.join(splitedLat)+'\n')
                    dataList.append([docID,label,sentence.strip(),splitedPre,splitedLat])
                else:
                    posNodoc+=1
            else:
                posNodoc+=1
print(negNodoc, posNodoc)
print(dataList[0])
print(dataList[-1])
with open('adeReconstructed.pkl','wb') as fp:
    pickle.dump(dataList, fp)

