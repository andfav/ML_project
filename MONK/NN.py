"""
Classe NeuralNetwork che implementa il nostro modello di rete neurale.
"""
from Input import OneOfKAttribute, Input, OneOfKInput, TRInput, OneOfKTRInput 
from ActivFunct import ActivFunct, Sigmoidal, Identity, SoftPlus, SymmetricSigmoidal
from DataSet import DataSet, ModeInput
from multiprocessing.dummy import Pool as ThreadPool
import math, time, random
import numpy as np
import matplotlib.pyplot as graphic
from numpy import linalg

from enum import Enum
class ModeLearn(Enum):
    BATCH = 1
    MINIBATCH = 2
    ONLINE = 3

class NeuralNetwork(object):
    
    #Da usare in fase di testing
    #weights[i]: np.array, matrice dei pesi della i+1-esima hidden unit
    #weights[-1]: np.array, matrice dei pesi delle output unit
    #weights[i][:,0] colonna dei pesi di bias 
    def __init__(self, trainingSet: list, Outputf : ActivFunct, new_hyp={}, weights:list = None, Hiddenf:ActivFunct = None):
        #Dizionario contenente i settaggi di default (ovviamente modificabili) 
        #degli iperparametri. 
        self.hyp = {'learnRate': 0.1,
                    'momRate':   0.1,
                    'regRate':   0.1,
                    'ValMax':    0.2,
                    'HiddenLayers': 1, 
                    'HiddenUnits':  2,
                    'OutputUnits':  1,
                    'MaxEpochs': 10e4,
                    'Tolerance': 10e-4,
                    'TauEpoch': 1,
                    'TauLearnRate': 0.1,
                    'VerboseFreq': 50}

        #Aggiornamento degli iperparametri.
        for key in new_hyp:
            if key in self.hyp:
                self.hyp[key] = new_hyp[key]
            else:
                raise ValueError ("new_hyp must be a subdict of hyp!")

        #Gestione delle funzioni di attivazione.
        self.Outputf = Outputf
        if Hiddenf == None:
            self.Hiddenf = Outputf
        else:
            self.Hiddenf = Hiddenf

        #Inserimento del training set.
        self.trainingSet = list()
        b1 = isinstance(trainingSet[0], TRInput)
        b2 = isinstance(trainingSet[0], OneOfKTRInput)
        b = b1 or b2
        if len(trainingSet) == 0 or not b:
            raise ValueError ("inserted TR set is not valid!")
        else:
            length = trainingSet[0].getLength()
            for el in trainingSet:
                b1 = isinstance(el, TRInput)
                b2 = isinstance(el, OneOfKTRInput)
                b = b1 or b2
                if not b or el.getLength() != length:
                    raise ValueError ("TR set not valid: not homogeneous")
                self.trainingSet.append(el)
            fanIn = length+1

        #Generazione casuale (o inserimento, se passati al costruttore) dei pesi relativi alle unità.
        if weights == None:
            #Generazione casuale dei pesi con l'uso del fanIn.
            #HiddenWeights/Bias: lista di array contenenti i pesi relativi agli HiddenLayers, si procede dall'interno
            #verso l'esterno.
            rang = self.hyp['ValMax']*2/fanIn

            #Creazione dei pesi degli HiddenLayers.
            self.HiddenWeights = list()
            self.HiddenBias = list()
            for i in range(self.hyp['HiddenLayers']):
                self.HiddenWeights.append(np.random.uniform(-rang,rang,(self.hyp['HiddenUnits'],length)))
                self.HiddenBias.append(np.random.uniform(-rang,rang,(self.hyp['HiddenUnits'],1)))
                if i == 0:
                    length = self.hyp['HiddenUnits']

            #Creazione dei pesi dell'OutputLayer.
            self.OutputWeights = np.random.uniform(-rang,rang,(self.hyp['OutputUnits'],self.hyp['HiddenUnits']))
            self.OutputBias = np.random.uniform(-rang,rang,(self.hyp['OutputUnits'],1))
        else:
            #Controllo validità dei pesi inseriti.
            if len(weights) != self.hyp['HiddenLayers']+1:
                raise ValueError ("NN, in __init__: wrong number of weights matrixes inserted")

            #Inserimento hidden weights e bias.
            for i in range(self.hyp['HiddenLayers']):
                if isinstance(weights[i], np.array):
                    if weights[i].shape == (self.hyp['HiddenUnits'],length+1):
                        self.HiddenWeights.append(weights[i][:,1:])
                        self.HiddenBias.append(np.array([weights[i][:,0]]).transpose())
                    else:
                        raise ValueError ("NN, in __init__: assigned hidden weights matrix not valid")

            #Inserimento output weights e bias.
            if isinstance(weights[-1], np.array):
                if weights[-1].shape == (self.hyp['OutputUnits'],self.hyp['HiddenUnits']+1):
                    self.OutputWeights = weights[-1][:,1:]
                    self.OutputBias = np.array([weights[-1][:,0]]).transpose()
                else:
                    raise ValueError ("NN, in __init__: assigned output weights matrix not valid")

    
    #Esegue backpropagation e istruisce la rete settando i pesi.
    def learn(self, mode:ModeLearn, errorFunct = None, numMiniBatches = None, validationSet = None, errorVlFunct= None, verbose: bool = False):
        if errorVlFunct == None:
            errorVlFunct = errorFunct

        momentum = self.hyp["momRate"]
        learnRateStart = self.hyp["learnRate"]
        regRate = self.hyp["regRate"]
        TauEpoch = self.hyp["TauEpoch"]
        TauLearnRate = self.hyp["TauLearnRate"]

        oldWeightsRatioOut = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        oldratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))

        oldWeightsRatioHidden = list()
        oldratio_Bias_hidden = list()
        for i in range(self.hyp['HiddenLayers']):
            dim1 = self.hyp['HiddenUnits']
            dim2 = self.hyp['HiddenUnits']
            if i == 0:
                dim2 = self.trainingSet[0].getLength()
            oldWeightsRatioHidden.append(np.zeros((dim1, dim2)))
            oldratio_Bias_hidden.append(np.zeros((dim1, 1)))

        #Inizializzazione di contatori di epochs ed iterazioni.
        epochs = 0
        it = 0

        #Inizializzazione delle liste degli errori/accuratezze.
        vecErr = list()
        vecAcc = list()
        vecVlErr = list()
        vecVlAcc = list()

        #Costruzione della funzione di accuratezza (solo per problemi di classificazione).
        if isinstance(self.Outputf,Sigmoidal):
            def accFunct(target,value):
                if value >= 0.5:
                    value = 0.9
                else:
                    value = 0.1
                return (target == value)
        else:
            accFunct = None
        
        #Inserimento dell'errore/accuratezza iniziale.
        err = self.getError(self.trainingSet,1/(len(self.trainingSet)),errorFunct)
        if accFunct != None:
            vecAcc.append(self.getError(self.trainingSet,1/(len(self.trainingSet)),accFunct))
        if validationSet != None:
            vlerr = self.getError(validationSet,1/(len(validationSet)),errorVlFunct)
            vecVlErr.append(vlerr)
            if accFunct != None:
                vecVlAcc.append(self.getError(validationSet,1/(len(validationSet)),accFunct))
        
        while(epochs < self.hyp["MaxEpochs"]  and err > self.hyp["Tolerance"]):
            if(epochs <= TauEpoch):
                alfa = epochs/TauEpoch
            learnRate = (1-alfa)*learnRateStart + alfa*TauLearnRate
            if mode == ModeLearn.BATCH:
                #Esecuzione di un'iterazione (l'intero data set).
                (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden) = self.batchIter(oldWeightsRatioOut, oldWeightsRatioHidden, oldratio_Bias_out, oldratio_Bias_hidden, learnRate, momentum, regRate)

                #Aggiornamento del contatore epochs e dell'errore/accuratezza sul trainingSet.
                epochs += 1
                err = self.getError(self.trainingSet,1/(len(self.trainingSet)),errorFunct,contr=False)
                if verbose and epochs % self.hyp['VerboseFreq'] == 0:
                    print("epoch: "+str(epochs))
                    print("errore: "+str(err))
                vecErr.append(err)
                if accFunct != None:
                    vecAcc.append(self.getError(self.trainingSet,1/(len(self.trainingSet)),accFunct,contr=False))

                #Aggiornamento dell'errore/accuratezza su un eventuale validationSet inserito.
                if validationSet != None:
                    vlerr = self.getError(validationSet,1/(len(validationSet)),errorVlFunct,contr=False)
                    vecVlErr.append(vlerr)
                    if accFunct != None:
                        vecVlAcc.append(self.getError(validationSet,1/(len(validationSet)),accFunct,contr=False))
                
            elif mode == ModeLearn.MINIBATCH:
                #Controllo inserimento del numero dei mini-batches e casting ad int.
                if numMiniBatches == None:
                    raise ValueError ("in learn: numMiniBatches not inserted.")
                numMiniBatches = int(numMiniBatches)

                if it % numMiniBatches == 0:
                    #Aggiornamento del contatore epochs e dell'errore/accuratezza.
                    if it != 0:
                        err = self.getError(self.trainingSet,1/(len(self.trainingSet)),errorFunct,contr=False)
                        vecErr.append(err)
                        if accFunct != None:
                            vecAcc.append(self.getError(self.trainingSet,1/(len(self.trainingSet)),accFunct,contr=False))

                        #Errore/accuratezza su un eventuale validationSet.
                        if validationSet != None:
                            vlerr = self.getError(validationSet,1/(len(validationSet)),errorVlFunct,contr=False)
                            vecVlErr.append(vlerr)
                            if accFunct != None:
                                vecVlAcc.append(self.getError(validationSet,1/(len(validationSet)),accFunct,contr=False))

                        epochs += 1
                        if verbose and epochs % self.hyp['VerboseFreq'] == 0:
                            print("epoch: "+str(epochs))
                            print("errore: "+str(err))
                    it = 0

                    #Rimescolamento del training set.
                    arr = self.trainingSet.copy()
                    random.shuffle(arr)

                    #Costruisco la sottolista dei dati divisibile esattamente per numMb.
                    h = len(arr) - len(arr) % numMiniBatches

                    #Creo la lista dei folders.
                    miniBatchDim = int(h / numMiniBatches)
                    miniBatches = [arr[i*miniBatchDim : (i+1)*miniBatchDim] for i in range(numMiniBatches)]

                    #Inserisco gli elementi di avanzo.
                    for i in range(len(arr)-h):
                        miniBatches[i].append(arr[i+h])

                #Esecuzione di un'iterazione (un mini-batch): il coefficiente di importanza è dato dal rapporto tra 
                #la lunghezza del mini-batch corrente e la lunghezza standard dei mini-batches.
                currentMiniBatch = miniBatches.pop(0)
                impCoeff = len(currentMiniBatch) / miniBatchDim
                (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden) = self.miniBatchIter(oldWeightsRatioOut, oldWeightsRatioHidden, oldratio_Bias_out, oldratio_Bias_hidden, currentMiniBatch, impCoeff, learnRate, momentum, regRate)
                it += 1
                
            elif mode == ModeLearn.ONLINE:
                if it % len(self.trainingSet) == 0:
                    #Aggiornamento del contatore epochs e dell'errore/accuratezza.
                    if it != 0:
                        epochs += 1
                        err = self.getError(self.trainingSet,1/(len(self.trainingSet)),errorFunct,contr=False)
                        if verbose and epochs % self.hyp['VerboseFreq'] == 0:
                            print("epoch: "+str(epochs))
                            print("errore: "+str(err))
                        vecErr.append(err)
                        if accFunct != None:
                            vecAcc.append(self.getError(self.trainingSet,1/(len(self.trainingSet)),accFunct,contr=False))

                        #Errore/accuratezza su un eventuale validationSet.
                        if validationSet != None:
                            vlerr = self.getError(validationSet,1/(len(validationSet)),errorVlFunct,contr=False)
                            vecVlErr.append(vlerr)
                            if accFunct != None:
                                vecVlAcc.append(self.getError(validationSet,1/(len(validationSet)),accFunct,contr=False))
                    it = 0

                    #Rimescolamento del training set.
                    arr = self.trainingSet.copy()
                    random.shuffle(arr)

                #Esecuzione di un'iterazione (un solo dato del training set).
                (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden) = self.onlineIter(oldWeightsRatioOut, oldWeightsRatioHidden, oldratio_Bias_out, oldratio_Bias_hidden, arr.pop(0), learnRate, momentum, regRate)
                it += 1

            #Aggiornamento degli old ratios.
            oldWeightsRatioOut = ratio_W_Out
            oldratio_Bias_out = ratio_Bias_out

            for i in range(self.hyp['HiddenLayers']):
                oldWeightsRatioHidden[i] = ratio_W_Hidden[i]
                oldratio_Bias_hidden[i] = ratio_Bias_hidden[i]

        return (vecErr,vecVlErr,vecAcc,vecVlAcc)
        

    
    #Se full == False, resituisce gli output di rete (array dei valori uscenti dalle unità di output) dato l'input inp;
    #se full == True, restituisce gli array di output e net di output layer, le liste di output e net degli hidden layers.
    def getOutput(self, inp:Input, full:bool=False):

        #Calcolo gli outputs (ed eventuali nets) delle hidden units
        hiddenNets = list()
        inp2hidd = list()
        inp2hidd.append(inp.getInput())
        for i in range(self.hyp['HiddenLayers']):
            hiddenNets.append(np.dot(self.HiddenWeights[i],inp2hidd[i]) + self.HiddenBias[i].transpose()[0])
            inp2hidd.append(self.Hiddenf.getf(hiddenNets[i]))

        #Calcolo gli outputs (ed eventuali nets) di rete
        outputNets = np.dot(self.OutputWeights,inp2hidd[-1]) + self.OutputBias.transpose()[0]
        hidd2out = self.Outputf.getf(outputNets)

        if full:
            return (hidd2out, outputNets, inp2hidd, hiddenNets)

        else:
            return hidd2out

        
        
    #Calcola l'errore (rischio) empirico della lista di TRInput o OneOfKTRInput data, con la funzione L (loss)
    # eventualmente assegnata, default=LMS e fattore di riscalamento k.
    def getError(self, data : list, k, L= None, contr=True):
        if L == None:
            L = lambda target,value: sum((target - value)**2)
        
        #Controllo di validità dei dati (se richiesto).
        if contr:
            b1 = isinstance(data[0], TRInput)
            b2 = isinstance(data[0], OneOfKTRInput)
            b = b1 or b2
            if len(data) == 0 or not b:
                raise ValueError ("inserted set is not valid!")
            else:
                length = data[0].getLength()
                for el in data:
                    b1 = isinstance(el, TRInput)
                    b2 = isinstance(el, OneOfKTRInput)
                    b = b1 or b2
                    if not b or el.getLength() != length:
                        raise ValueError ("data set not valid: not homogeneous")
        
        #Calcolo effettivo dell'errore.
        s = 0
        for d in data:
            if isinstance(self.Outputf, Sigmoidal):
                target = d.getTargetSigmoidal()
            else:
                target = d.getTarget()
            s += L(target,self.getOutput(d))
        return k*s

    """
    Metodo che, dato un input con target, calcola tutti i delta necessari alla back-prop relativi
    al suddetto input.
    """
    def getDeltas(self, inp: Input):
        b1 = isinstance(inp, TRInput)
        b2 = isinstance(inp, OneOfKTRInput)

        if b1 or b2:
            #Calcolo gli outputs di rete, le nets dell'outputLayer e degli hiddenLayer.
            (Outputs,OutputNet,OutputsHidden,HiddenNets) = self.getOutput(inp,full=True)

            #Calcolo i delta relativi alle output units.
            deltaOutResults = (inp.getTarget() - Outputs) * self.Outputf.getDerivative(OutputNet)

            #Calcolo i delta relativi alle hidden units. (La notazione l[::-1] inverte la posizione degli elementi in una lista)
            HiddenNets = HiddenNets[::-1]
            deltaHiddenResults = list()
            deltaHiddenResults.append(deltaOutResults)
            currentWeights = self.OutputWeights
            for i in range(self.hyp['HiddenLayers']):
                deltaHiddenResults.append(np.dot(deltaHiddenResults[i],currentWeights) * self.Hiddenf.getDerivative(HiddenNets[i]))
                currentWeights = self.HiddenWeights[self.hyp['HiddenLayers']-1-i]
            deltaHiddenResults = deltaHiddenResults[1:][::-1]

        else:
            raise ValueError ("in getDeltas: inserted non-targeted input.")

        return (OutputsHidden, deltaOutResults, deltaHiddenResults)

    """
    Metodo che implementa una iterazione dell'algoritmo batch backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: lista (per hidden layer) di np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioHidden: lista (per hidden layer) np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
        learnRate: learning rate
        momRate: alfa del momentum
        regRate: lambda della regolarizzazione

    -return
        (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden): variazioni calcolate sull'iterazione corrente
    """

    def batchIter(self, oldWeightsRatioOut: np.array, oldWeightsRatioHidden: np.array, oldBiasRatioOut: np.array, oldBiasRatioHidden: np.array, learnRate, momRate, regRate):

        ratio_W_Out = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))

        ratio_W_Hidden = list()
        ratio_Bias_hidden = list()
        for i in range(self.hyp['HiddenLayers']):
            dim1 = self.hyp['HiddenUnits']
            dim2 = self.hyp['HiddenUnits']
            if i == 0:
                dim2 = self.trainingSet[0].getLength()
            ratio_W_Hidden.append(np.zeros((dim1, dim2)))
            ratio_Bias_hidden.append(np.zeros((dim1, 1)))

        #scorro sugli input
        for inp in self.trainingSet:
            #Calcolo dei valori delta.
            (hiddenOutResults,deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
            #Aggiornamento dei ratios relativi a pesi e bias dello hidden layer.
            for i in range(self.hyp['HiddenLayers']):
                inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([hiddenOutResults[i]]))
                ratio_W_Hidden[i] += learnRate*np.dot(np.diag(deltaHiddenResults[i]),inputsMatrix)
                ratio_Bias_hidden[i] += learnRate*np.array([deltaHiddenResults[i]]).transpose()

            #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
            hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults[-1]]))
            ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
            ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
            
        #Riscalamento del gradiente.
        ratio_W_Out /= len(self.trainingSet)
        ratio_Bias_out /= len(self.trainingSet)
        for i in range(self.hyp['HiddenLayers']):
            ratio_W_Hidden[i] /= len(self.trainingSet)
            ratio_Bias_hidden[i] /= len(self.trainingSet)

        #Aggiunta della componente momentum.
        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_Bias_out += momRate * oldBiasRatioOut
        for i in range(self.hyp['HiddenLayers']):
            ratio_W_Hidden[i] += momRate * oldWeightsRatioHidden[i]
            ratio_Bias_hidden[i] += momRate * oldBiasRatioHidden[i]

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        for i in range(self.hyp['HiddenLayers']):
            self.HiddenWeights[i] = self.HiddenWeights[i] + ratio_W_Hidden[i] - regRate*self.HiddenWeights[i]
            self.HiddenBias[i] = self.HiddenBias[i] + ratio_Bias_hidden[i]

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)
    
    """
    Metodo che implementa una iterazione dell'algoritmo mini-batch backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: lista (per hidden layer) di np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioHidden: lista (per hidden layer) di np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
        minBatch: mini-batch corrente
        impCoeff: coefficiente di importanza del mini-batch corrente in relazione alla sua dimensione rispetto
                  agli altri mini-batches 
        learnRate: learning rate
        momRate: alfa del momentum
        regRate: lambda della regolarizzazione

    -return
        (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden): variazioni calcolate sull'iterazione corrente
    """

    def miniBatchIter(self, oldWeightsRatioOut: np.array, oldWeightsRatioHidden: np.array, oldBiasRatioOut: np.array, oldBiasRatioHidden: np.array, minBatch: list, impCoeff: float, learnRate, momRate, regRate):

        ratio_W_Out = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))

        ratio_W_Hidden = list()
        ratio_Bias_hidden = list()
        for i in range(self.hyp['HiddenLayers']):
            dim1 = self.hyp['HiddenUnits']
            dim2 = self.hyp['HiddenUnits']
            if i == 0:
                dim2 = self.trainingSet[0].getLength()
            ratio_W_Hidden.append(np.zeros((dim1, dim2)))
            ratio_Bias_hidden.append(np.zeros((dim1, 1)))

        #scorro sugli input
        for inp in minBatch:
            #Calcolo dei valori delta.
            (hiddenOutResults, deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
            #Aggiornamento dei ratios relativi a pesi e bias degli hidden layers.
            for i in range(self.hyp['HiddenLayers']):
                inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([hiddenOutResults[i]]))
                ratio_W_Hidden[i] += learnRate*np.dot(np.diag(deltaHiddenResults[i]),inputsMatrix)
                ratio_Bias_hidden[i] += learnRate*np.array([deltaHiddenResults[i]]).transpose()

            #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
            hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults[-1]]))
            ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
            ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
        
        #Riscalamento del gradiente.
        ratio_W_Out /= len(minBatch)
        ratio_Bias_out /= len(minBatch)
        for i in range(self.hyp['HiddenLayers']):
            ratio_W_Hidden[i] /= len(minBatch)
            ratio_Bias_hidden[i] /= len(minBatch)

        #Aggiunta della componente momentum.
        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_Bias_out += momRate * oldBiasRatioOut
        for i in range(self.hyp['HiddenLayers']):
            ratio_W_Hidden[i] += momRate * oldWeightsRatioHidden[i]
            ratio_Bias_hidden[i] += momRate * oldBiasRatioHidden[i]

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        for i in range(self.hyp['HiddenLayers']):
            self.HiddenWeights[i] = self.HiddenWeights[i] + ratio_W_Hidden[i] - regRate*self.HiddenWeights[i]
            self.HiddenBias[i] = self.HiddenBias[i] + ratio_Bias_hidden[i]

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)

    """
    Metodo che implementa una iterazione dell'algoritmo online backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: lista (per hidden layer) degli np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioHidden: lista (per hidden layer) np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
        inp: input corrente
        learnRate: learning rate
        momRate: alfa del momentum
        regRate: lambda della regolarizzazione

    -return
        (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden): variazioni calcolate sull'iterazione corrente
    """

    def onlineIter(self, oldWeightsRatioOut: np.array, oldWeightsRatioHidden: np.array, oldBiasRatioOut: np.array, oldBiasRatioHidden: np.array, inp, learnRate, momRate, regRate):

        ratio_W_Out = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))

        ratio_W_Hidden = list()
        ratio_Bias_hidden = list()
        for i in range(self.hyp['HiddenLayers']):
            dim1 = self.hyp['HiddenUnits']
            dim2 = self.hyp['HiddenUnits']
            if i == 0:
                dim2 = self.trainingSet[0].getLength()
            ratio_W_Hidden.append(np.zeros((dim1, dim2)))
            ratio_Bias_hidden.append(np.zeros((dim1, 1)))

        #Calcolo dei valori delta.
        (hiddenOutResults,deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
        #Aggiornamento dei ratios relativi a pesi e bias degli hidden layers.
        for i in range(self.hyp['HiddenLayers']):
                inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([hiddenOutResults[i]]))
                ratio_W_Hidden[i] += learnRate*np.dot(np.diag(deltaHiddenResults[i]),inputsMatrix)
                ratio_Bias_hidden[i] += learnRate*np.array([deltaHiddenResults[i]]).transpose()

        #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
        hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults[-1]]))
        ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
        ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
        
        #Aggiunta della componente momentum
        ratio_W_Out = (1 - momRate)*ratio_W_Out + momRate * oldWeightsRatioOut
        ratio_Bias_out = (1-momRate)*ratio_Bias_out + momRate * oldBiasRatioOut
        for i in range(self.hyp['HiddenLayers']):
            ratio_W_Hidden[i] = (1 - momRate)*ratio_W_Hidden[i] + momRate * oldWeightsRatioHidden[i]
            ratio_Bias_hidden[i] = (1-momRate)*ratio_Bias_hidden[i] + momRate * oldBiasRatioHidden[i]

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        for i in range(self.hyp['HiddenLayers']):
            self.HiddenWeights[i] = self.HiddenWeights[i] + ratio_W_Hidden[i] - regRate*self.HiddenWeights[i]
            self.HiddenBias[i] = self.HiddenBias[i] + ratio_Bias_hidden[i]

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)

    """
    Metodo per il plotting veloce di grafici in serie usando matplotlib.
    -arguments
        vecList: lista di tuple di numpy.array, ogni tupla contiene i grafici
        da inserire nella singola figura.
        opt: lista di tuple di stringhe, ogni tupla contiene le opzioni di
        plotting nei grafici da inserire nella singola figura.
    """
    def getPlot(self,vecList:list, opt:list):
        if len(vecList) != len(opt):
            raise ValueError ("In getPlot: vecList and opt dim mismatch.")
        else:
            for el in vecList:
                i = vecList.index(el)
                if len(el) != len(opt[i]):
                    raise ValueError ("In getPlot: number of vects and options mismatch.")
                else:
                    for v in el:
                        j = el.index(v)
                        graphic.plot(v,opt[i][j])
                graphic.show()


#Test.
"""
f = Sigmoidal(12)

domains = [-1, 3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = [1]

trainingSet = DataSet("monks-1.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
testSet = DataSet("monks-1.test", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)

neruale = NeuralNetwork(trainingSet.inputList, f, {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.7, 'regRate':0, 'Tolerance':0.0001, 'MaxEpochs': 800})
start = time.time()
(errl, errtr, accl, acctr) = neruale.learn(ModeLearn.BATCH,validationSet=testSet.inputList,numMiniBatches=5, verbose=False)
end = time.time()
print("tempo: "+str(end - start))
neruale.getPlot([[errl,errtr],[accl, acctr]],[('r','--'),('r','--')])
"""
"""
xorSet = DataSet("Xor.txt", " ", ModeInput.TR_INPUT, 3) 
xorNN = NeuralNetwork(xorSet.inputList, f, {'HiddenUnits':4, 'learnRate':0.025, 'ValMax':0.7, 'momRate':0.1, 'regRate':0.001, 'Tolerance':0.001, 'MaxEpochs': 9000})
errList = xorNN.learn(ModeLearn.ONLINE)
xorNN.getPlot(errList)
xorTest = DataSet("Xor.txt", " ", ModeInput.INPUT, columnSkip=[3]) 

print("0 0 : "+ str(xorNN.getOutput(xorTest.inputList[0])))
print("0 1 : "+ str(xorNN.getOutput(xorTest.inputList[1])))
print("1 0 : "+ str(xorNN.getOutput(xorTest.inputList[2])))
print("1 1 : "+ str(xorNN.getOutput(xorTest.inputList[3])))
"""
"""
s = 0
for d in testSet.inputList:
    out = (neruale.getOutput(d)[0] >= 0.5)
    s += abs(out - d.getTarget())
perc = 1 - s/len(testSet.inputList)
print("Accuratezza sul test set: " + str(perc*100) + "%.")
"""

def MEE(target,output):
    return linalg.norm(target - output,2)

outputF = Identity()
hiddenf = SymmetricSigmoidal(10,2)
skipRow = [1,2,3,4,5,6,7,8,9,10]
columnSkip = [1]
target = [12,13]

trSet = DataSet("ML-CUP18-TR.csv", ",", ModeInput.TR_INPUT, target, None, skipRow, columnSkip)

random.shuffle(trSet.inputList)
tSet = trSet.inputList[0:9*len(trSet.inputList)//10]
vlSet = trSet.inputList[9*len(trSet.inputList)//10:]
cupNN = NeuralNetwork(tSet, outputF, {'OutputUnits':2, 'HiddenLayers':2, 'HiddenUnits':20, 'learnRate':0.005, 'TauEpoch':1, 'TauLearnRate':0.005, 'ValMax':1, 'momRate':0.5, 'regRate':1e-10, 'Tolerance':0.01, 'MaxEpochs': 25000}, Hiddenf=hiddenf)

"""
cupNN = NeuralNetwork(trSet.inputList, outputF, {'OutputUnits':2, 'HiddenLayers':2, 'HiddenUnits':100, 'learnRate':0.005, 'TauEpoch':1, 'TauLearnRate':0.0005, 'ValMax':1, 'momRate':0.7, 'regRate':0, 'Tolerance':0.01, 'MaxEpochs': 100000}, Hiddenf=hiddenf)
"""

print("Learning...")
start = time.time()
(errtr, errvl, acctr, accvl) = cupNN.learn(ModeLearn.BATCH, validationSet=vlSet, errorFunct=MEE, verbose=True)

"""
(errtr, errvl, acctr, accvl) = cupNN.learn(ModeLearn.BATCH, errorFunct=MEE, verbose=True)
"""

end = time.time()
print("Errore ultima epoch: " + str(errtr[-1]))
print("Tempo: "+str(end-start))

out = open('mlcupresSS10,2_HL2_HU20_LR5e-3_RG1e-10_EP25k', 'a')
for el in errtr:
    out.write(str(el)+",")
out.write("\n")
out.close()

out = open('mlcupresSS10,2_HL2_HU20_LR5e-3_RG1e-10_EP25k[vlSet]', 'a')
for el in errvl:
    out.write(str(el)+",")
out.write("\n")
out.close()

graphic.axis((0,25000,0,2))
graphic.grid()
graphic.title('SymmetricSigmoidal10,2 HU:20x2HL LR:5e-3 mom:0.5 RG:1e-10')
graphic.plot(errtr,color='blue',label='training')
graphic.plot(errvl,'--',color='red',label='test')
graphic.legend()
graphic.show()


