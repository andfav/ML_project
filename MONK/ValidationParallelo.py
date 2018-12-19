from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
from multiprocessing import Pool
from functools import partial
import random as rnd
import time



"""
Funzione per un insieme di iperparametri fissati
param:
    k: numero di folder
    dataset: insieme di trSet+vlSet
    f: funzione di attivazione
    theta: dizionario contenente gli iperparametri
    errorFunct: funzione che stima l'errore
    modeLearn: Online/MiniBatch/Batch
    miniBatchDim: dimensione di ciascun minibatch
"""
def k_fold_CV_single(k: int, dataSet, f:ActivFunct, theta, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if len(dataSet) % k == 0:
        rnd.shuffle(dataSet)

        folderDim = int(len(dataSet) / k)
        folder = [dataSet[i*folderDim : (i+1)*folderDim] for i in range(k)]
        errore = list()
        pool = Pool(k)
        poolList = list()
        for i in range (len(folder)):
            lcopy = folder.copy()
            del(lcopy[i])

            vlSet = folder[i]
            trSet = list()
            for j in range(len(lcopy)):
                trSet+= lcopy[j]
            nn = NeuralNetwork(trSet, f, theta)
            poolList.append((nn,vlSet))
        
        errore = pool.map(partial(task_cv_single,modeLearn=modeLearn, errorFunct=errorFunct, miniBatchDim=miniBatchDim),poolList)
            
        err = sum(errore)/k
        return err
    else:
        raise ValueError("k must divide data set length")

def task_cv_single(t, modeLearn: ModeLearn, errorFunct:ActivFunct, miniBatchDim = None):
    (nn, vlSet) = t
    nn.learn(modeLearn, errorFunct, miniBatchDim)
    e = nn.getError(vlSet, 0, 1/len(vlSet), errorFunct)
    return e

f = Sigmoidal(8)

domains = [3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = 1

trainingSet = DataSet("monks-1.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
theta = {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.001, 'MaxEpochs': 600}
start = time.time()
e = k_fold_CV_single(4, trainingSet.inputList, f, theta, None,  ModeLearn.BATCH, None)
stop = time.time() 
secdiff = stop - start
print("errore: "+str(e)+" tempo: "+str(secdiff))