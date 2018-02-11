import scipy
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVC

class DataLoader(object):
    def __init__(self,dir='data/'):
        self.rawData=pd.read_csv(dir+'train.csv')
        self.testData=pd.read_csv(dir+'test.csv')
        self.dataTransformMap={
            'COLLEGE':{
                'zero':0,
                'one':1
            },
            'REPORTED_SATISFACTION':{
                'very_unsat':0,
                'unsat':25,
                'avg':50,
                'sat':75,
                'very_sat':100
            },
            'REPORTED_USAGE_LEVEL':{
                'very_little':0,
                'little':1,
                'avg':2,
                'high':3,
                'very_high':4
            },
            'CONSIDERING_CHANGE_OF_PLAN':{
                'considering':0,
                'actively_looking_into_it':1,
                'perhaps':2,
                'no':3,
                'never_thought':4
            }
        }

        self.rawData.replace(inplace=True,to_replace=self.dataTransformMap)
        self.testData.replace(inplace=True,to_replace=self.dataTransformMap)

        self.features=list(self.rawData.columns)

    def split(self,ratio=0.9):
        '''
        split the dataset (should be rawData) into train and val set, will shuffle the dataset each time called
        :param ratio: ratio of train/val, 0.9 by default
        :return: trainX,trainy,valX,valy
        '''
        n=self.rawData.__len__()
        self.rawData.sample(frac=1).reset_index(drop=True)
        trainNum=int(n*ratio)
        valNum=n-trainNum
        tx=self.rawData.head(trainNum)[self.features]
        ty=self.rawData.head(trainNum)['LEAVE']
        vx=self.rawData.tail(valNum)[self.features]
        vy=self.rawData.tail(valNum)['LEAVE']

        return tx,ty,vx,vy


class Classifier(object):
    def __init__(self):
        dataLoader=DataLoader()
        self.trainX,self.trainY,self.valX,self.valY=dataLoader.split()
        self.clf=SVC(probability=True)
    def run(self):
        self.clf.fit(self.trainX,self.trainY)

    def predict(self):
        predY=self.clf.predict(self.valX)

        return predY


    def evaluate(self,predY):
        acc=sk.metrics.accuracy_score(self.valY,predY)
        return acc

c=Classifier()
c.run()
res=c.predict()
acc=c.evaluate(res)
print(acc)
