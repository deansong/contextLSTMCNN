import glob
import sys
import gensim
import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import label_binarize
import random
import logging
from sklearn.feature_extraction.text import CountVectorizer

class FOFEW2V:
    def __init__(self):
        self.sentence = []
        self.w2vModel = None
        self.maxSentLen = 50
        self.ebd_size = 50
        self.num_iter = 20
        self.wordAlpha = 0.95
        
    def setLogging(self, logLevel='info'):
        loggingLevels={'debug': logging.DEBUG,
                      'info': logging.INFO}
        logging.basicConfig(level=loggingLevels[logLevel])


    def read_sent(self,dirname):
        for subFolder in glob.glob(dirname+'/*'):
            print(subFolder)
            for txtFile in glob.glob(subFolder+'/*.txt'):
                print(txtFile)
                for line in open(txtFile,'r'):
                    self.sentence.append(line.split())

    def w2v_training_sents(self, labeledList, idlist):
        x=[]
        for i in idlist:
            doc = labeledList[i]
            for line in doc[2]:
                x.append(line[1].split())
        return x


    def build_w2v(self):
        self.w2vModel = gensim.models.Word2Vec(self.sentence, min_count=1, iter=self.num_iter,size=self.ebd_size)

    def save_w2v(self,model2save):
        self.w2vModel.save(model2save)

    def load_w2v(self, model2load):
        self.w2vModel = gensim.models.Word2Vec.load(model2load)

    def sentW2v(self, inputSent, ebd_size):
        alpha = self.wordAlpha
        w2vList = []
        fofeCode=np.array([0]*ebd_size)
        i=0
        for word in inputSent:
            try:
                currentW2V = self.w2vModel[word]
            except:
                currentW2V = np.array([0]*ebd_size)
            w2vList.append(currentW2V.tolist())
            fofeCode = fofeCode*alpha + currentW2V
            i+=1
            if i == self.maxSentLen:
                break
        while(i< self.maxSentLen):
            w2vList.append([0]*ebd_size)
            i+=1
        return w2vList, fofeCode

    def w2vList2fofe(self,w2vlist,alpha=0.95):
        fofeCode = None
        fofeStart = 0
        for w2v in w2vlist:
            w2v = np.array(w2v)
            if fofeStart == 1:
                fofeCode = fofeCode*alpha + w2v
            else:
                fofeCode = w2v
                fofeStart = 1
        return fofeCode

    def fofe(self, inputSent, alpha=0.95):
        fofeCode = None
        fofeStart = 0
        for word in inputSent:
            if word in self.w2vModel:
                w2v = self.w2vModel[word]
                if fofeStart == 1:
                    fofeCode = fofeCode*alpha + w2v
                else:
                    fofeCode = w2v
                    fofeStart = 1
        return fofeCode,fofeStart


class DataProcess(FOFEW2V):
    def __init__(self):
        FOFEW2V.__init__(self)
        self.avilableLabels=['noLabel', 'Suicidality', 'Non-suicidal', 'Uncertain']
        self.vocab={} #word = [id]
        self.inverseVocab = []

    def buildDict(self, topWords=2000):
        from collections import Counter
        wordId = 0
        wordcount = Counter(self.sentence)
        for word, _ in wordcount.most_common(topWords):
            self.vocab[word] = wordId
            self.inverseVocab.append(word)
            wordId+=1

    def sent2bow(self, labeledList, idlist):
        print(len(idlist))
        x=[]
        y=[]
        total = [0] * len(self.avilableLabels)
        vectorizer = CountVectorizer()
        vectorizer.vocabulary = self.inverseVocab
        print(idlist)
        for i in idlist:
            print(i)
            doc = labeledList[i]
            for line in doc[2]:
                currentLabel = line[0]
                if currentLabel in self.avilableLabels:
                    idx = self.avilableLabels.index(currentLabel)
                    currentbow = vectorizer.fit_transform([line[1]]).toarray().tolist()[0]
                    total[idx] +=1
                    x.append(currentbow)
                    y.append(idx)
        return x,y,total


    def doc2W2Vdoc(self, labeledList, idlist, oriSent=False):
        x=[]
        y=[]
        if oriSent:
            orix=[]
        else:
            orix=None
        all_sents=[]
        total = [0] * len(self.avilableLabels)
        for i in idlist:
            doc = labeledList[i]
            currentOriListx = []
            currentDocListx = []
            currentDocListy = []
            for line in doc[2]:
                currentLabel = line[0]
                if currentLabel in self.avilableLabels:
                    idx = self.avilableLabels.index(currentLabel)
                    binLabel = label_binarize([line[0]], self.avilableLabels).tolist()
                    w2vList,fofeCode = self.sentW2v(line[1].split(), self.ebd_size)
                    total[idx] +=1
                    currentDocListx.append([w2vList, fofeCode])
                    currentDocListy.append(binLabel[0])
                    if oriSent:
                        currentOriListx.append(line[1])        
                else:
                    logging.debug(currentLabel +' not in avilable label list, ignored')
                
            y.append(currentDocListy)
            x.append(currentDocListx)
            if oriSent:
                orix.append(currentOriListx)
        return x,y,orix, total

    def w2vdoc2fofesentWindow(self,x,y, oriX=None, windowSize=99999):
        totalDoc = len(x)
        display = 100
        for i in range(len(x)):
            if i % display == 0:
                print('processed',i, totalDoc)
            for j in range(len(x[i])):
                currentLabel = y[i][j]
                currentx= x[i][j][0]
                if j == 0:
                    prex = [0]*self.ebd_size
                else:
                    preHead = j - windowSize
                    if preHead > 0:
                        prex = self.sent_fofe_window(x[i][preHead:j])
                    else:
                        self.sent_fofe_window(x[i][0:j])
                if (j+1) == len(x[i]):
                    latx = [0]*self.ebd_size
                else:
                    preTail = j + 1 + windowSize
                    if preTail <= len(x[i]):
                        latx = self.sent_fofe_window(x[i][j+1:preTail], reverse = True)
                    else:
                        latx = self.sent_fofe_window(x[i][j+1:], reverse = True)
                if oriX:
                    currentorix = oriX[i][j]
                    yield currentx, prex, latx, currentorix, currentLabel
                else:
                    yield currentx, prex, latx, currentLabel



    def sampleData(self, cyid, pyid, lyid, thres):
        selected = False
        if cyid == 0:
            if pyid != 0 or lyid != 0:
                selected = True
            else:
                r = random.random()
                if r < thres:
                    selected = True
                else:
                    selected = False
        else:
            selected = True
        return selected



    def w2vdoc2fofesentWindowSample(self,x,y, oriX=None, windowSize = 99999, thres=0.01, useAll=False, reduced=False):
        totalDoc = len(x)
        display = 100
        for i in range(len(x)):
            if i % display == 0:
                print('processed',i, totalDoc)
            for j in range(len(x[i])):
                selected =False
                currentx = None
                prex = None
                latx = None
                currentLabel = None
                currentorix = None
                currentLabel = y[i][j]
                currentx= x[i][j][0]
                yid = currentLabel.index(1)
                if oriX:
                    currentorix = oriX[i][j]
                yid = currentLabel.index(1)
                if j == 0:
                    preLabel = [0,0,0,0]
                    preyid = None
                else:
                    preLabel =  y[i][j-1]
                    preyid = preLabel.index(1)

                if (j+1) == len(x[i]):
                    latLabel = [0,0,0,0]
                    latyid = None
                else:
                    latLabel = y[i][j+1]
                    latyid = latLabel.index(1)


                if useAll == False:
                    if reduced:
                        selected = self.sampleData(yid, preyid, latyid, thres)
                    else:
                        selected = self.sampleData(yid, preyid, latyid, thres)
                else:
                    selected = True    

                if selected == True:
                    if j == 0:
                        prex = [0]*self.ebd_size
                    else:
                        preHead = j - windowSize
                        if preHead > 0:
                            prex = self.sent_fofe_window(x[i][preHead:j])
                        else:
                            prex = self.sent_fofe_window(x[i][0:j])
                    if (j+1) == len(x[i]):
                        latx = [0]*self.ebd_size
                    else:
                        preTail = j + 1 + windowSize
                        if preTail <= len(x[i]):
                            latx = self.sent_fofe_window(x[i][j+1:preTail], reverse = True)
                        else:
                            latx = self.sent_fofe_window(x[i][j+1:], reverse = True)
                yield currentx, prex, latx, currentorix, currentLabel, preLabel, latLabel, selected


    def sent_fofe_window(self, doc_w2v_list, reverse = False, alpha = 0.1):
        start = 0
        #fofeCode = None
        if reverse == True:
            doc_w2v_list.reverse()
        for sent_list in doc_w2v_list:
            currentfofe = sent_list[1]
            if start == 0:
                fofeCode = currentfofe
                start = 1
            else:
                try:
                    fofeCode = fofeCode*alpha + currentfofe
                except:
                    print(currentfofe, fofeCode)
        return fofeCode.tolist()


class DataProcess2class(DataProcess):
    def __init__(self):
        DataProcess.__init__(self)
        self.avilableLabels=['Suicidality', 'Non-suicidal']


    def doc2W2Vdoc(self, labeledList, idlist, oriSent=False):
        x=[]
        y=[]
        orix=[]
        all_sents=[]
        total = [0] * len(self.avilableLabels)
        for i in idlist:
            doc = labeledList[i]
            currentOriListx = []
            currentDocListx = []
            currentDocListy = []
            for line in doc[2]:
                currentLabel = line[0]
                w2vList,fofeCode = self.sentW2v(line[1].split(), self.ebd_size)
                currentDocListx.append([w2vList, fofeCode])
                if oriSent:
                    currentOriListx.append(line[1])

                if currentLabel in self.avilableLabels:
                    idx = self.avilableLabels.index(currentLabel)
                    binLabel = label_binarize([line[0]], self.avilableLabels).tolist()
                    total[idx] +=1
                    currentDocListy.append(binLabel[0])
                else:
                    currentDocListy.append('ignore')
            y.append(currentDocListy)
            x.append(currentDocListx)
            orix.append(currentOriListx)
        return x,y,orix, total


    def w2vdoc2fofesentWindowSample(self,x,y, oriX=None, windowSize = 99999, useAll=False):
        totalDoc = len(x)
        display = 100
        for i in range(len(x)):
            if i % display == 0:
                print('processed',i, totalDoc)
            for j in range(len(x[i])):
                selected =False
                currentx = None
                prex = None
                latx = None
                currentLabel = None
                currentorix = None
                preLabel = None
                latLabel = None
                currentLabel = y[i][j]
                currentx= x[i][j][0]
                if currentLabel != 'ignore':
                    selected = True
                    if oriX:
                        currentorix = oriX[i][j]
                    if j == 0:
                        prex = [0]*self.ebd_size
                    else:
                        preHead = j - windowSize
                        if preHead > 0:
                            prex = self.sent_fofe_window(x[i][preHead:j])
                        else:
                            prex = self.sent_fofe_window(x[i][0:j])
                    if (j+1) == len(x[i]):
                        latx = [0]*self.ebd_size
                    else:
                        preTail = j + 1 + windowSize
                        if preTail <= len(x[i]):
                            latx = self.sent_fofe_window(x[i][j+1:preTail], reverse = True)
                        else:
                            latx = self.sent_fofe_window(x[i][j+1:], reverse = True)
                yield currentx, prex, latx, currentorix, currentLabel, preLabel, latLabel, selected



if __name__ == "__main__":
    sentence = FOFEW2V()
    sentence.read_sent(sys.argv[1])
    sentence.build_w2v()
    print(sentence.w2vModel['patient'])
    print(sentence.w2vModel.most_similar(positive=['patient']))
    print('sdfsdfdsfdsfdsfdsfdsfdsfsdf' in sentence.w2vModel)
    testff ='I am writing to kindly let you know that I saw  ZZZZZ  along with his mother on 27 November to complete his assessment.'.split()
    ff=sentence.fofe(testff)
    print(ff)
