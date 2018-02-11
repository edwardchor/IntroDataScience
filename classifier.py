import scipy
import numpy as np
import pandas as pd
import sklearn as sk

class DataLoader(object):
    def __init__(self,dir='data/'):
        self.rawData=pd.read_csv(dir+'train.csv')
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
        
