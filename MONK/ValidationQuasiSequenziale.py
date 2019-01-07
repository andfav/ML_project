from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random as rnd
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as graphic

def k_fold_CV_single(k: int, dataSet, f:ActivFunct, theta, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if k <= 0:
        raise ValueError ("Wrong value of num. folders inserted")

    cp = dataSet.copy()
    
    #Rimescolo il data set.
    rnd.shuffle(cp)

    #Costruisco la sottolista dei dati divisibile esattamente per k.
    h = len(cp) - len(cp) % k
    dataSetExact = cp[0:h]

    #Creo la lista dei folders.
    folderDim = int(len(dataSetExact) / k)
    folders = [cp[i*folderDim : (i+1)*folderDim] for i in range(k)]

    #Inserisco gli elementi di avanzo.
    for i in range(len(cp)-h):
        folders[i].append(cp[i+h])

    errore = list()

    #per stampare l'errore sul traininig set e sul validation set
    trErrorPlot = list()
    vlErrorPlot = list()

    for i in range (len(folders)):
        lcopy = folders.copy()
        del(lcopy[i])

        vlSet = folders[i]
        trSet = list()
        for j in range (len (lcopy)):
            trSet+= lcopy[j]
        nn = NeuralNetwork(trSet, f, theta)
        (trErr, vlErr, trAcc, vlAcc) = nn.learn(modeLearn, errorFunct, miniBatchDim, vlSet)
        trErrorPlot.append(trErr)
        vlErrorPlot.append(vlErr)
        errore.append(nn.getError(vlSet, 0, 1/len(vlSet), errorFunct))

    err = sum(errore)/k
    trErrorArray = np.array(trErrorPlot[0])
    vlErrorArray = np.array(vlErrorPlot[0])

    for i in range (1, len(trErrorPlot)):
        trErrorArray = trErrorArray + np.array(trErrorPlot[i])
    trErrorArray = trErrorArray / k

    for i in range (1, len(vlErrorPlot)):
        vlErrorArray = vlErrorArray + np.array(vlErrorPlot[i])
    vlErrorArray = vlErrorArray / k
    
    return (err, trErrorArray, vlErrorArray)

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
    trErrorPlot = list()
    vlErrorPlot = list()
    for i in range(0, nIter):
        (err, trErr, vlErr) = k_fold_CV_single(nFolder,dataSet,f,theta,errorFunct,modeLearn,miniBatchDim)
        errore.append(err)
        trErrorPlot.append(trErr)
        vlErrorPlot.append(vlErr)

    endTime = time.time()
    print("************")
    print("Configurazione completata, tempo: " + str(endTime - startTime))
    print(str((theta, sum(errore)/nIter)))

    trArray = trErrorPlot[0]
    vlArray = vlErrorPlot[0]

    for i in range (1,nIter):
        trArray = trArray + trErrorPlot[i]
        vlArray = vlArray + vlErrorPlot[i]

    trArray = trArray / nIter
    vlArray = vlArray / nIter

    return (theta, sum(errore)/nIter, trArray, vlArray)

"""
Funzione che implementa la double cross validation.
Argomenti:
- workers, numero massimo di processi attivi;
- testFolder, numero di folder in cui dividere inizialmente il dataSet (testSet e validation + training);
- nFolder, numero di folder di ciascuna k-fold validation;
- dataSet, il data set;
- f, funzione di attivazione della rete neurale;
- theta, diziorario di tutti gli iperparametri;
- errorFunct, funzione di valutazione dell'errore;

def double_cross_validation(workers: int, testFolder: int, nFolder: int, dataSet, f:ActivFunct, theta:dict, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, nIter: int = 10):
     if testFolder <= 1:
        raise ValueError ("Wrong value of num. folders inserted")
    
    #Rimescolo il data set.
    rnd.shuffle(dataSet.copy())

    #Costruisco la sottolista dei dati divisibile esattamente per testFolder.
    h = len(dataSet) - len(dataSet) % testFolder
    dataSetExact = dataSet[0:h]

    #Creo la lista dei folders.
    folderDim = int(len(dataSetExact) / testFolder)
    folders = [dataSet[i*folderDim : (i+1)*folderDim] for i in range(testFolder)]

    #Inserisco gli elementi di avanzo.
    for i in range(len(dataSet)-h):
        folders[i].append(dataSet[i+h])

    for i in range(len(folders)):
        testSet = folders[i]
"""

def getBestResult(e):
    l = [e[i][1] for i in range(len(e))]
    return e[l.index(min(l))]

if __name__ == '__main__':
    f = Sigmoidal(12)

    domains = [3, 3, 2, 3, 4, 2]
    columnSkip = [8]
    targetPos = 1
    datasetFileName = "monks-1.train"
    logFileName = datasetFileName + "sigm12finalregular3.log"

    trainingSet = DataSet(datasetFileName, " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
    learnRates = [0.1]
    momRates = [0.5, 0.6, 0.7]
    regRates = [0.0019, 0.0018]
    ValMaxs = [0.75]
    HiddenUnitss = [6]
    OutputUnitss = [1]
    MaxIterss = [800]
    Tolerances = [0.0001]
    start = time.time()
    e = cross_validation(6, 12, ModeLearn.BATCH, trainingSet.getInputs(), f, None, learnRates, momRates, regRates, ValMaxs, HiddenUnitss, OutputUnitss, MaxIterss, Tolerances, start, None)
    stop = time.time() 
    secdiff = stop - start

    log = open(logFileName,'a')
    log.write("***\n")
    log.write("File: " + datasetFileName + ", con configurazioni di iperparametri seguenti \n")
    log.write("learnRates: " + str(learnRates))
    log.write('\n')
    log.write("momRates: " + str(momRates))
    log.write('\n')
    log.write("regRates: " + str(regRates))
    log.write('\n')
    log.write("ValMaxs: " + str(ValMaxs))
    log.write('\n')
    log.write("HiddenUnitss: " + str(HiddenUnitss))
    log.write('\n')
    log.write("OutputUnitss: " + str(OutputUnitss))
    log.write('\n')
    log.write("MaxIterss: " + str(MaxIterss))
    log.write('\n')
    log.write("Tolerances: " + str(Tolerances))
    log.write('\n\n')

    log.write("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi, risultati:\n")

    log.write(str(e) + '\n')
    log.write('\n')
    log.write("Miglior risultato: " + str(getBestResult(e)) + '\n')
    log.write('\n\n')

    log.close()

    for elem in e:
        graphic.plot(elem[2], 'r--')#tr set
        graphic.plot(elem[3], 'b')#vl set
        graphic.show()

    print("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi: i dati sono stati salvati in " + logFileName + ".")