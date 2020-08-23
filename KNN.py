import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from point2d import Point2D
from numpy import log as ln
import sys
import math
import collections
from math import *
from data import *
from decimal import Decimal
from scipy.spatial import distance


def knn ( base:data,test:data,p:float,k:float)->(float):
    errors=0
    # row 0-the min dis
    # row 1 is the gender
    kNeighbors=np.full((2,k),sys.float_info.max)
    for check in test:
        for data in base:
            dis = distance.minkowski(check.asVecrtorP(),data.asVecrtorP(),p)
            uppdateNewNeighbors(kNeighbors,dis,data.gender)
        # sum which gender is the winner
        num1 = collections.Counter(kNeighbors[1]).get(1)
        if (num1 == None):
            num1 = 0
        num2 = collections.Counter(kNeighbors[1]).get(2)
        if (num2 == None):
            num2 = 0
        # if we classifier wrong
        if ((num1>num2)and(check.gender==2) or (num2>num1)and (check.gender==1)):
            errors +=1
        kNeighbors = np.full((2, k), sys.float_info.max)

    return errors

def uppdateNewNeighbors(kNeighbors:np.ndarray,newdis:float,gender:float):
    num_rows, num_cols = kNeighbors.shape
    maxValue=kNeighbors[0].max()
    if(maxValue>newdis):
        for i in range (num_cols):
            if (kNeighbors[0][i] ==maxValue):
                kNeighbors[0][i]=newdis
                kNeighbors[1][i]=gender
                break



def main():
#loading the dataset
    dataload = open("HC_Body_Temperature.txt", "r")
    dataload=dataload.read().splitlines()
    dataSet=np.ndarray(130,dtype=data)
    for i in range(0,130):
        temperature, gender, heartrate = dataload[i].split()
        newData=data(float(temperature),float(heartrate),float(gender))
        dataSet[i]=newData
    test=np.full((3,15),0,float)
    j=0
    Trainerrors=0
    Testerrors=0
    for p in ([1, 2, inf]):
        print("---------------------------")
        print("p - ",p)
        for k in ([1, 3, 5, 7, 9]):
            for i in range(500):
                np.random.shuffle(dataSet)
                train_data = dataSet[:65]
                test_data = dataSet[65:]
                Trainerrors=knn(train_data,train_data,p,k)
                Testerrors+=knn(train_data,test_data,p,k)
            test[0][j]=(Testerrors/500)/65
            if(p==inf):
                test[1][j]=-1
            else:
                test[1][j] = p
            test[2][j]=k
            j+=1
            print("********k = ", k ,"******")
            print(" the errors of the test ", (Testerrors/500)/65,"%")
            print(" the errors of the train", (Trainerrors/500)/65,"%")
            Testerrors=0
            Trainerrors=0

    minValue=test[0].min()
    for i in range(j):
        if(test[0][i] == minValue):
            if (test[1][i] == -1):
                print("Best knn to get the min errors is for p inf ", "and for k " , test[2][i], ". \n the error is: ", test[0][i])
            else:
                print ("Best knn to get the min errors is for p ",test[1][i],"and for k ",test[2][i],". \n the error is: ",test[0][i])








if __name__ == '__main__':
    main()








