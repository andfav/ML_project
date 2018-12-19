from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
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
        for i in range (len(folder)):
            lcopy = folder.copy()
            del(lcopy[i])

            vlSet = folder[i]
            trSet = list()
            for j in range (len (lcopy)):
                trSet+= lcopy[j]
            nn = NeuralNetwork(trSet, f, theta)
            nn.learn(modeLearn, errorFunct, miniBatchDim)
            errore.append(nn.getError(vlSet, 0, 1/len(vlSet), errorFunct))

        err = sum(errore)/k
        return err
    else:
        raise ValueError("k must divide data set length")

f = Sigmoidal(8)

domains = [3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = 1

trainingSet = DataSet("monks-2.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
theta = {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.0001, 'MaxEpochs': 600}
start = time.time()
e = k_fold_CV_single(13, trainingSet.inputList, f, theta, None,  ModeLearn.BATCH, None)
stop = time.time() 
secdiff = stop - start
print("errore: "+str(e)+" tempo: "+str(secdiff))