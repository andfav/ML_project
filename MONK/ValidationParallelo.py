from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random as rnd
import numpy as np
from numpy.linalg import norm
import time
import itertools



"""
Funzione per un insieme di iperparametri fissati
param:
    workers: grado di parallelismo
    k: numero di folder
    dataset: insieme di trSet+vlSet
    f: funzione di attivazione
    theta: dizionario contenente gli iperparametri
    errorFunct: funzione che stima l'errore
    modeLearn: Online/MiniBatch/Batch
    miniBatchDim: dimensione di ciascun minibatch
"""
def k_fold_CV_single(workers: int, k: int, dataSet, f:ActivFunct, theta:dict, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if workers <= 0:
        workers = 1

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
        
    poolList = list()
    with ProcessPoolExecutor(max_workers=min(k, workers)) as pool:
        for i in range (len(folders)):
            lcopy = folders.copy()
            del(lcopy[i])

             #Creo validation e training set.
            vlSet = folders[i]
            trSet = list()
            for j in range(len(lcopy)):
                trSet+= lcopy[j]
                
            #Aggiorno la poolList.
            poolList.append((trSet,vlSet))
        
        #In parallelo creo, istruisco le reti, calcolo gli errori sui possibili folders.
        errore = list(pool.map(partial(task_cv_single,modeLearn=modeLearn, f=f, theta=theta, errorFunct=errorFunct, miniBatchDim=miniBatchDim),poolList))

    #Restituisco l'errore medio.    
    err = sum(errore)/k
    return err

"""
Task che viene eseguito in parallelo dai processi creati da k_fold_CV_single
"""
def task_cv_single(t, modeLearn: ModeLearn, f:ActivFunct, theta:dict, errorFunct:ActivFunct, miniBatchDim = None):
    (trSet, vlSet) = t
    nn = NeuralNetwork(trSet, f, theta)
    nn.learn(modeLearn, errorFunct, miniBatchDim)

    vecErr = np.array([nn.getError(vlSet, i, 1/len(vlSet), errorFunct) for i in range(nn.hyp['OutputUnits'])])
    return norm(vecErr,2)

"""
Implementazione della K-Fold-Cross-Validation su un insieme di iperparametri
param:
    workers: grado di parallelismo
    nFolder: numero di folder della cross validation
    modeLearn: Online/MiniBatch/Batch
    dataset: insieme di trSet+vlSet
    f: funzione di attivazione
    errorFunct: funzione che stima l'errore
    learnRate: lista di tutti i possibili valori che il learning rate può assumere all'interno della cross validation
    momRate: lista di tutti i possibili valori che il momentum rate può assumere all'interno della cross validation
    regRate: lista di tutti i possibili valori che il regularization rate può assumere all'interno della cross validation
    ValMax: lista di tutti i possibili valori che il parametro ValMax (valore massimo dei pesi) della rete neurale può assumere all'interno della cross validation
    HiddenUnits: lista di tutti i possibili valori che il parametro HiddenUnits (numero di unità nell'hidden layer) della rete neurale può assumere all'interno della cross validation
    OutputUnits: lista di tutti i possibili valori che il parametro OutputUnits (numero di unità di output) della rete neurale può assumere all'interno della cross validation
    MaxEpochs: lista di tutti i possibili valori che il parametro MaxEpochs (numero massimo di epoche) della rete neurale può assumere all'interno della cross validation
    Tolerance: lista di tutti i possibili valori che il parametro Tolerance della rete neurale può assumere all'interno della cross validation
    miniBatchDim: dimensione di ciascun minibatch (se modeLearn = Minibatch)
"""
def cross_validation(workers: int, nFolder:int, modeLearn:ModeLearn, dataSet, f:ActivFunct, errorFunct:ActivFunct, learnRate:list, momRate:list, regRate:list, ValMax:list, HiddenUnits:list, OutputUnits:int, MaxEpochs:list, Tolerance:list, startTime, miniBatchDim= None):
    if workers <= 0:
        workers = 1
    
    dictionaries = list()
 
    #generazione di tutte le possibili combinazioni
    for maxVal in ValMax:
        for alfa in momRate:
            for lambd in regRate:
                for eta in learnRate:
                    for hUnit in HiddenUnits:
                        for epochs in MaxEpochs:
                            for epsilon in Tolerance:                                
                                theta = {'learnRate':eta, 'momRate':alfa, 'regRate':lambd, 'ValMax':maxVal, 'HiddenUnits':hUnit, 'OutputUnits':OutputUnits, 'MaxEpochs':epochs, 'Tolerance':epsilon}
                                dictionaries.append(theta)

        totalTheta = len(dictionaries)
        print("Configurazioni totali di iperparametri: "+str(totalTheta))

        res = list()
        future = list()
        with ProcessPoolExecutor(max_workers=6) as pool:
            for theta in dictionaries:
                future.append(pool.submit(partial(cross_validation_iterator, nIter=10, workers=workers, nFolder=nFolder, dataSet=dataSet.copy(), f=f, theta=theta, startTime=startTime, errorFunct=errorFunct, modeLearn=modeLearn, miniBatchDim=miniBatchDim)))

            for elem in future:
                res.append(elem.result())

    return res
    
def cross_validation_iterator(workers: int, nFolder: int, dataSet, f:ActivFunct, theta:dict, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, nIter: int = 10):
    future = list()
    with ProcessPoolExecutor(max_workers=min(workers,nIter)) as pool:
        for i in range(0, nIter):
            future.append(pool.submit(partial(k_fold_CV_single, workers, k=nFolder, dataSet=dataSet, modeLearn=modeLearn, f=f, theta=theta, errorFunct=errorFunct, miniBatchDim=miniBatchDim)))

        
        errore = list()
        for elem in future:
            errore.append(elem.result())

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
    datasetFileName = "monks-1.train"
    logFileName = datasetFileName + ".log"

    trainingSet = DataSet(datasetFileName, " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
    theta = {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.037, 'MaxEpochs': 600}
    start = time.time()
    e = cross_validation(60, 8, ModeLearn.BATCH, trainingSet.getInputs(), f, None, [0.1, 0.075, 0.05, 0.025, 0.01], [0.7, 0.6, 0.5, 0.4, 0.3], [0.01, 0.005, 0], [0.7], [2, 3, 4, 5], 1, [1000], [0.006], start, None)
    stop = time.time() 
    secdiff = stop - start

    log = open(logFileName,'a')
    log.write("***\n")
    log.write(str(e) + '\n')
    log.write('\n')
    log.write("Miglior risultato: " + str(getBestResult(e)) + '\n')
    log.write('\n\n')

    print("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi: i dati sono stati salvati in " + logFileName + ".")