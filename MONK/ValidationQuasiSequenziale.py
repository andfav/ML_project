from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random as rnd
import numpy as np
from numpy.linalg import norm
import time

def k_fold_CV_single(k: int, dataSet, f:ActivFunct, theta, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if k <= 0:
        raise ValueError ("Wrong value of num. folders inserted")
    
    #Rimescolo il data set.
    rnd.shuffle(dataSet.copy())

    #Costruisco la sottolista dei dati divisibile esattamente per k.
    h = len(dataSet) - len(dataSet) % k
    dataSetExact = dataSet[0:h]

    #Creo la lista dei folders.
    folderDim = int(len(dataSetExact) / k)
    folders = [dataSet[i*folderDim : (i+1)*folderDim] for i in range(k)]

    #Inserisco gli elementi di avanzo.
    for i in range(len(dataSet)-h):
        folders[i].append(dataSet[i+h])

    errore = list()
    for i in range (len(folders)):
        lcopy = folders.copy()
        del(lcopy[i])

        vlSet = folders[i]
        trSet = list()
        for j in range (len (lcopy)):
            trSet+= lcopy[j]
        nn = NeuralNetwork(trSet, f, theta)
        nn.learn(modeLearn, errorFunct, miniBatchDim)
        errore.append(nn.getError(vlSet, 0, 1/len(vlSet), errorFunct))

    err = sum(errore)/k
    return err

def cross_validation(workers: int, nFolder:int, modeLearn:ModeLearn, dataSet, f:ActivFunct, errorFunct:ActivFunct, learnRate:list, momRate:list, regRate:list, ValMax:list, HiddenUnits:list, OutputUnits:list, MaxEpochs:list, Tolerance:list, startTime, miniBatchDim= None):
    if workers <= 0:
        workers = 1
    
    dictionaries = list()
 
    #generazione di tutte le possibili combinazioni
    for eta in learnRate:
        for alfa in momRate:
            for lambd in regRate:
                for maxVal in ValMax:
                    for hUnit in HiddenUnits:
                        for oUnit in OutputUnits:
                            for epochs in MaxEpochs:
                                for epsilon in Tolerance:
                                    theta = {'learnRate':eta, 'momRate':alfa, 'regRate':lambd, 'ValMax':maxVal, 'HiddenUnits':hUnit, 'OutputUnits':oUnit, 'MaxEpochs':epochs, 'Tolerance':epsilon}
                                    dictionaries.append(theta)

    totalTheta = len(dictionaries)
    print("Configurazioni totali di iperparametri: "+str(totalTheta))

    res = list()
    future = list()
    with ProcessPoolExecutor(max_workers=min(workers,totalTheta)) as pool:
        for theta in dictionaries:
            future.append(pool.submit(partial(cross_validation_iterator, nIter=10, workers=workers, nFolder=nFolder, dataSet=dataSet.copy(), f=f, theta=theta, startTime=startTime, errorFunct=errorFunct, modeLearn=modeLearn, miniBatchDim=miniBatchDim)))

        for elem in future:
            res.append(elem.result())

    return res
    
def cross_validation_iterator(workers: int, nFolder: int, dataSet, f:ActivFunct, theta:dict, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, nIter: int = 10):
    errore = list()
    for i in range(0, nIter):
        errore.append(k_fold_CV_single(nFolder,dataSet,f,theta,errorFunct,modeLearn,miniBatchDim))

    endTime = time.time()
    print("************")
    print("Configurazione completata, tempo: " + str(endTime - startTime))
    print(str((theta, sum(errore)/nIter)))

    return (theta, sum(errore)/nIter)

def getBestResult(e):
    l = [e[i][1] for i in range(len(e))]
    return e[l.index(min(l))]

if __name__ == '__main__':
    f = Sigmoidal(8)

    domains = [3, 3, 2, 3, 4, 2]
    columnSkip = [8]
    targetPos = 1
    datasetFileName = "monks-3.train"
    logFileName = datasetFileName + ".log"

    trainingSet = DataSet(datasetFileName, " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
    theta = {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.037, 'MaxEpochs': 600}
    start = time.time()
    e = cross_validation(300, 10, ModeLearn.BATCH, trainingSet.getInputs(), f, None, [0.1, 0.05, 0.01], [0.7, 0.6, 0.5], [0.1, 0.01, 0], [0.7], [4, 5], [1], [600], [0.001], start, None)
    stop = time.time() 
    secdiff = stop - start

    log = open(logFileName,'a')
    log.write("***\n")
    log.write(str(e) + '\n')
    log.write('\n')
    log.write("Miglior risultato: " + str(getBestResult(e)) + '\n')
    log.write('\n\n')

    print("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi: i dati sono stati salvati in " + logFileName + ".")