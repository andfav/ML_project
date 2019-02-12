from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity, SymmetricSigmoidal
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random as rnd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as graphic
import time
import itertools
from numpy import linalg



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
def k_fold_CV_single(workers: int, k: int, dataSet, f:ActivFunct, theta:dict, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, errorVlFunct = None, hiddenF: ActivFunct=None):
    if workers <= 0:
        workers = 1

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
        
    

    future = list()
    with ProcessPoolExecutor(max_workers=min(k, workers)) as pool:
        for i in range (len(folders)):
            lcopy = folders.copy()
            del(lcopy[i])

            #Creo validation e training set.
            vlSet = folders[i]
            trSet = list()
            for j in range(len(lcopy)):
                trSet+= lcopy[j]
                
            #In parallelo creo, istruisco le reti, calcolo gli errori sui possibili folders.
            future.append(pool.submit(partial(task_cv_single, t= (trSet, vlSet), modeLearn=modeLearn, f=f, theta=theta, errorFunct=errorFunct, miniBatchDim=miniBatchDim, errorVlFunct=errorVlFunct, hiddenF=hiddenF)))
            
        for elem in future:
            (trError, vlError, error) = (elem.result())
            print("DEBUG k_fold_cv_single: trError: "+str(trError))
            trErrorPlot.append(trError)
            vlErrorPlot.append(vlError)
            errore.append(error)

    err = sum(errore)/k

    #controllo che tutti gli errorPlot abbiano la stessa lunghezza
    maxLen = len(trErrorPlot[0])
    for i in range(1, len(trErrorPlot)):
        if len(trErrorPlot[i]) > maxLen:
            maxLen = len(trErrorPlot[i])

    for i in range(len(trErrorPlot)):
        if len(trErrorPlot[i]) < maxLen:
            for j in range(maxLen-len(trErrorPlot[i])):
                trErrorPlot[i].append(trErrorPlot[i][-1])
                vlErrorPlot[i].append(vlErrorPlot[i][-1])


    trErrorArray = np.array(trErrorPlot[0])
    vlErrorArray = np.array(vlErrorPlot[0])

    
    for i in range (1, len(trErrorPlot)):
        trErrorArray = trErrorArray + np.array(trErrorPlot[i])
    trErrorArray = trErrorArray / k

    for i in range (1, len(vlErrorPlot)):
        vlErrorArray = vlErrorArray + np.array(vlErrorPlot[i])
    vlErrorArray = vlErrorArray / k
    
    return (err, trErrorArray, vlErrorArray)

"""
Task che viene eseguito in parallelo dai processi creati da k_fold_CV_single
"""
def task_cv_single(t, modeLearn: ModeLearn, f:ActivFunct, theta:dict, errorFunct:ActivFunct, miniBatchDim = None, errorVlFunct = None, hiddenF: ActivFunct=None):
    (trSet, vlSet) = t
    
    nn = NeuralNetwork(trSet, f, theta, Hiddenf=hiddenF)
   
    (trErr, vlErr, _, _) = nn.learn(modeLearn, errorFunct, miniBatchDim, vlSet, errorVlFunct=errorVlFunct)
    print("DEBUG task_cv_single")

    errore = (nn.getError(vlSet, 1/len(vlSet), errorVlFunct))
    return (trErr, vlErr, errore)

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
def cross_validation(workers: int, nFolder:int, modeLearn:ModeLearn, dataSet, f:ActivFunct, errorFunct:ActivFunct, learnRate:list, momRate:list, regRate:list, ValMax:list, HiddenUnits:list, OutputUnits:list, MaxEpochs:list, Tolerance:list, startTime, miniBatchDim= None, errorVlFunct = None, hiddenF: ActivFunct=None):
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
                                    theta = {'learnRate':eta, 'momRate':alfa, 'regRate':lambd, 'ValMax':maxVal, 'HiddenUnits':hUnit, 'OutputUnits':oUnit, 'MaxEpochs':epochs, 'Tolerance':epsilon, 'TauEpoch':1, 'TauLearnRate':eta}
                                    dictionaries.append(theta)

    totalTheta = len(dictionaries)
    print("Configurazioni totali di iperparametri: "+str(totalTheta))

    res = list()
    future = list()
    with ProcessPoolExecutor(max_workers=totalTheta) as pool:
        for theta in dictionaries:
            future.append(pool.submit(partial(cross_validation_iterator, nIter=10, workers=workers, nFolder=nFolder, dataSet=dataSet.copy(), f=f, theta=theta, startTime=startTime, errorFunct=errorFunct, modeLearn=modeLearn, miniBatchDim=miniBatchDim, errorVlFunct=errorVlFunct, hiddenF=hiddenF)))

        for elem in future:
            res.append(elem.result())

    return res
    
def cross_validation_iterator(workers: int, nFolder: int, dataSet, f:ActivFunct, theta:dict, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, nIter: int = 10, errorVlFunct = None, hiddenF: ActivFunct=None):
    errore = list()
    trErrorPlot = list()
    vlErrorPlot = list()
    future = list()
    with ProcessPoolExecutor(max_workers=min(workers,nIter)) as pool:
        for i in range(0, nIter):
            future.append(pool.submit(partial(k_fold_CV_single, 10, k=nFolder, dataSet=dataSet, modeLearn=modeLearn, f=f, theta=theta, errorFunct=errorFunct, miniBatchDim=miniBatchDim, errorVlFunct=errorVlFunct, hiddenF=hiddenF)))
      
        for elem in future:
            (trErr, vlErr, error) = elem.result()
            trErrorPlot.append(trErr.toList())
            vlErrorPlot.append(vlErr.toList())
            errore.append(error)


    endTime = time.time()
    print("************")
    print("Configurazione completata, tempo: " + str(endTime - startTime))
    print(str((theta, sum(errore)/nIter)))

    #controllo che tutti gli errorPlot abbiano la stessa lunghezza
    maxLen = len(trErrorPlot[0])
    for i in range(1, len(trErrorPlot)):
        if len(trErrorPlot[i]) > maxLen:
            maxLen = len(trErrorPlot[i])

    for i in range(len(trErrorPlot)):
        if len(trErrorPlot[i]) < maxLen:
            for j in range(maxLen-len(trErrorPlot[i])):
                trErrorPlot[i].append(trErrorPlot[i][-1])
                vlErrorPlot[i].append(vlErrorPlot[i][-1])

    
    trArray = np.array(trErrorPlot[0])
    vlArray = np.array(vlErrorPlot[0])

    for i in range (1,nIter):
        trArray = trArray + np.array(trErrorPlot[i])
        vlArray = vlArray + np.array(vlErrorPlot[i])

    trArray = trArray / nIter
    vlArray = vlArray / nIter

    return (theta, sum(errore)/nIter, trArray, vlArray)

def getBestResult(e):
    l = [e[i][1] for i in range(len(e))]
    return e[l.index(min(l))]

def MEE(target,output):
    return linalg.norm(target - output,2)

if __name__ == '__main__':
    print("Validation...")
    outputF = Identity()
    hiddenf = SymmetricSigmoidal(80,2)
    skipRow = [1,2,3,4,5,6,7,8,9,10]
    columnSkip = [1]
    target = [12,13]

    trSet = DataSet("Z:\Matteo\Desktop\Machine Learning\ML-CUP18-TR.csv", ",", ModeInput.TR_INPUT, target, None, skipRow, columnSkip)
    
    learnRates = [5e-3]
    momRates = [0.5]
    regRates = [1e-10]
    ValMaxs = [1]
    HiddenUnitss = [300]
    OutputUnitss = [2]
    MaxIterss = [25000]
    Tolerances = [0.01]
    start = time.time()
    e = cross_validation(7, 7, ModeLearn.BATCH, trSet.inputList, outputF, MEE, learnRates, momRates, regRates, ValMaxs, HiddenUnitss, OutputUnitss, MaxIterss, Tolerances, start, None, errorVlFunct=MEE, hiddenF=hiddenf)
    stop = time.time() 
    secdiff = stop - start

    datasetFileName = "Z:\Matteo\Desktop\Machine Learning\ML-CUP18-TR.csv"
    logFileName = datasetFileName + ".log"
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
        graphic.title('SymmetricSigmoidal80,2'+ 'HU:'+str(HiddenUnitss[0])+' LR:'+str(learnRates[0])+' mom:'+str(momRates[0])+' Regular:'+str(regRates[0]))
        graphic.plot(elem[2], 'r--', label='trset')#tr set
        graphic.plot(elem[3], 'b', label='vlset')#vl set
        graphic.show()
    
    
    print("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi: i dati sono stati salvati in " + logFileName + ".")