"""
Classe NeuralNetwork che implementa il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
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
    #weights[0]: np.array, matrice dei pesi delle hidden unit
    #weights[1]: np.array, matrice dei pesi delle output unit
    #weights[:,0] colonna dei pesi di bias 
    def __init__(self, trainingSet: list, Outputf : ActivFunct, new_hyp={}, weights:list = None, Hiddenf:ActivFunct = None):
        #Dizionario contenente i settaggi di default (ovviamente modificabili) 
        #degli iperparametri. 
        self.hyp = {'learnRate': 0.1,
                    'momRate':   0.1,
                    'regRate':   0.1,
                    'ValMax':    0.2,
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
            rang = self.hyp['ValMax']*2/fanIn 
            self.HiddenWeights = np.random.uniform(-rang,rang,(self.hyp['HiddenUnits'],length))
            self.HiddenBias = np.random.uniform(-rang,rang,(self.hyp['HiddenUnits'],1))
            self.OutputWeights = np.random.uniform(-rang,rang,(self.hyp['OutputUnits'],self.hyp['HiddenUnits']))
            self.OutputBias = np.random.uniform(-rang,rang,(self.hyp['OutputUnits'],1))
        else:
            #Inserimento hidden weights e bias.
            if isinstance(weights[0], np.array):
                if weights[0].shape == (self.hyp['HiddenUnits'],length+1):
                    self.HiddenWeights = weights[0][:,1:]
                    self.HiddenBias = np.array([weights[0][:,0]]).transpose()
                else:
                    raise ValueError ("NN, in __init__: assigned hidden weights matrix not valid")

            #Inserimento output weights e bias.
            if isinstance(weights[1], np.array):
                if weights[1].shape == (self.hyp['OutputUnits'],self.hyp['HiddenUnits']+1):
                    self.OutputWeights = weights[1][:,1:]
                    self.OutputBias = np.array([weights[0][:,0]]).transpose()
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
        oldWeightsRatioHidden = np.zeros((self.hyp['HiddenUnits'], self.trainingSet[0].getLength()))
        oldratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))
        oldratio_Bias_hidden = np.zeros((self.hyp['HiddenUnits'], 1))

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

            oldWeightsRatioHidden = ratio_W_Hidden
            oldratio_Bias_hidden = ratio_Bias_hidden

        return (vecErr,vecVlErr,vecAcc,vecVlAcc)
        

    
    #Se full == False, resituisce gli output di rete (array dei valori uscenti dalle unità di output) dato l'input inp;
    #se full == True, restituisce gli array di output e net di output layer, output e net di hidden layer.
    def getOutput(self, inp:Input, full:bool=False):

        #Calcolo gli outputs (ed eventuali nets) delle hidden units
        hiddenNets = np.dot(self.HiddenWeights,inp.getInput()) + self.HiddenBias.transpose()[0]
        inp2hidd = self.Hiddenf.getf(hiddenNets)

        #Calcolo gli outputs (ed eventuali nets) di rete
        outputNets = np.dot(self.OutputWeights,inp2hidd) + self.OutputBias.transpose()[0]
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
            #Calcolo gli outputs di rete, le nets dell'outputLayer e hiddenLayer.
            (Outputs,OutputNet,OutputsHidden,HiddenNet) = self.getOutput(inp,full=True)

            #Calcolo i delta relativi alle output units.
            deltaOutResults = (inp.getTarget() - Outputs) * self.Outputf.getDerivative(OutputNet)

            #Calcolo i delta relativi alle hidden units.
            deltaHiddenResults = np.dot(deltaOutResults,self.OutputWeights) * self.Hiddenf.getDerivative(HiddenNet)

        else:
            raise ValueError ("in getDeltas: inserted non-targeted input.")

        return (OutputsHidden, deltaOutResults, deltaHiddenResults)

    """
    Metodo che implementa una iterazione dell'algoritmo batch backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
        learnRate: learning rate
        momRate: alfa del momentum
        regRate: lambda della regolarizzazione

    -return
        (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden): variazioni calcolate sull'iterazione corrente
    """

    def batchIter(self, oldWeightsRatioOut: np.array, oldWeightsRatioHidden: np.array, oldBiasRatioOut: np.array, oldBiasRatioHidden: np.array, learnRate, momRate, regRate):

        ratio_W_Out = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        ratio_W_Hidden = np.zeros((self.hyp['HiddenUnits'], self.trainingSet[0].getLength()))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))
        ratio_Bias_hidden = np.zeros((self.hyp['HiddenUnits'], 1))

        #scorro sugli input
        for inp in self.trainingSet:
            #Calcolo dei valori delta.
            (hiddenOutResults, deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
            #Aggiornamento dei ratios relativi a pesi e bias dello hidden layer.
            inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([inp.getInput()]))
            ratio_W_Hidden += learnRate*np.dot(np.diag(deltaHiddenResults),inputsMatrix)
            ratio_Bias_hidden += learnRate*np.array([deltaHiddenResults]).transpose()

            #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
            hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults]))
            ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
            ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
            
        ratio_W_Out /= len(self.trainingSet)
        ratio_W_Hidden /= len(self.trainingSet)
        ratio_Bias_out /= len(self.trainingSet)
        ratio_Bias_hidden /= len(self.trainingSet)

        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_W_Hidden += momRate * oldWeightsRatioHidden
        ratio_Bias_out += momRate * oldBiasRatioOut
        ratio_Bias_hidden += momRate * oldBiasRatioHidden

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        self.HiddenWeights = self.HiddenWeights + ratio_W_Hidden - regRate*self.HiddenWeights
        self.HiddenBias = self.HiddenBias + ratio_Bias_hidden

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)
    
    """
    Metodo che implementa una iterazione dell'algoritmo mini-batch backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
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
        ratio_W_Hidden = np.zeros((self.hyp['HiddenUnits'], self.trainingSet[0].getLength()))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))
        ratio_Bias_hidden = np.zeros((self.hyp['HiddenUnits'], 1))

        #scorro sugli input
        for inp in minBatch:
            #Calcolo dei valori delta.
            (hiddenOutResults, deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
            #Aggiornamento dei ratios relativi a pesi e bias dello hidden layer.
            inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([inp.getInput()]))
            ratio_W_Hidden += learnRate*np.dot(np.diag(deltaHiddenResults),inputsMatrix)
            ratio_Bias_hidden += learnRate*np.array([deltaHiddenResults]).transpose()

            #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
            hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults]))
            ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
            ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
            
        ratio_W_Out /= len(minBatch)
        ratio_W_Hidden /= len(minBatch)
        ratio_Bias_out /= len(minBatch)
        ratio_Bias_hidden /= len(minBatch)

        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_W_Hidden += momRate * oldWeightsRatioHidden
        ratio_Bias_out += momRate * oldBiasRatioOut
        ratio_Bias_hidden += momRate * oldBiasRatioHidden

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        self.HiddenWeights = self.HiddenWeights + ratio_W_Hidden - regRate*self.HiddenWeights
        self.HiddenBias = self.HiddenBias + ratio_Bias_hidden

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)

    """
    Metodo che implementa una iterazione dell'algoritmo online backprop

    -arguments
        oldWeightsRatioOut: np.array delle variazioni dei pesi delle unità di output all'iterazione precedente
        oldWeightsRatioHidden: np.array delle variazioni dei pesi delle unità hidden all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità di output all'iterazione precedente
        oldBiasRatioOut: np.array delle variazioni dei bias delle unità hidden all'iterazione precedente
        inp: input corrente
        learnRate: learning rate
        momRate: alfa del momentum
        regRate: lambda della regolarizzazione

    -return
        (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden): variazioni calcolate sull'iterazione corrente
    """

    def onlineIter(self, oldWeightsRatioOut: np.array, oldWeightsRatioHidden: np.array, oldBiasRatioOut: np.array, oldBiasRatioHidden: np.array, inp, learnRate, momRate, regRate):

        ratio_W_Out = np.zeros((self.hyp['OutputUnits'], self.hyp['HiddenUnits']))
        ratio_W_Hidden = np.zeros((self.hyp['HiddenUnits'], self.trainingSet[0].getLength()))
        ratio_Bias_out = np.zeros((self.hyp['OutputUnits'], 1))
        ratio_Bias_hidden = np.zeros((self.hyp['HiddenUnits'], 1))

        #Calcolo dei valori delta.
        (hiddenOutResults,deltaOutResults,deltaHiddenResults) = self.getDeltas(inp)
        
        #Aggiornamento dei ratios relativi a pesi e bias dello hidden layer.
        inputsMatrix = np.dot(np.ones((self.hyp['HiddenUnits'],1)),np.array([inp.getInput()]))
        ratio_W_Hidden += learnRate*np.dot(np.diag(deltaHiddenResults),inputsMatrix)
        ratio_Bias_hidden += learnRate*np.array([deltaHiddenResults]).transpose()

        #Aggiornamento dei ratios relativi a pesi e bias dello output layer.
        hiddOutputsMatrix = np.dot(np.ones((self.hyp['OutputUnits'],1)),np.array([hiddenOutResults]))
        ratio_W_Out += learnRate*np.dot(np.diag(deltaOutResults),hiddOutputsMatrix)
        ratio_Bias_out += learnRate*np.array([deltaOutResults]).transpose()
        
        ratio_W_Out = (1 - momRate)*ratio_W_Out + momRate * oldWeightsRatioOut
        ratio_W_Hidden = (1 - momRate)*ratio_W_Hidden + momRate * oldWeightsRatioHidden
        ratio_Bias_out = (1-momRate)*ratio_Bias_out + momRate * oldBiasRatioOut
        ratio_Bias_hidden = (1-momRate)*ratio_Bias_hidden + momRate * oldBiasRatioHidden

        #aggiornamento pesi unità di output
        self.OutputWeights = self.OutputWeights + ratio_W_Out - regRate*self.OutputWeights
        self.OutputBias = self.OutputBias + ratio_Bias_out

        #aggiornamento pesi unità hidden layer
        self.HiddenWeights = self.HiddenWeights + ratio_W_Hidden - regRate*self.HiddenWeights
        self.HiddenBias = self.HiddenBias + ratio_Bias_hidden

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
hiddenf = SoftPlus()
skipRow = [1,2,3,4,5,6,7,8,9,10]
columnSkip = [1]
target = [12,13]

trSet = DataSet("ML-CUP18-TR.csv", ",", ModeInput.TR_INPUT, target, None, skipRow, columnSkip)
cupNN = NeuralNetwork(trSet.inputList, outputF, {'OutputUnits':2, 'HiddenUnits':70, 'learnRate':0.01, 'TauEpoch':20000, 'TauLearnRate':0.0001, 'ValMax':1, 'momRate':0.5, 'regRate':0, 'Tolerance':0.5, 'MaxEpochs': 60000}, Hiddenf=hiddenf)
print("Learning...")
start = time.time()
(errtr, errvl, acctr, accvl) = cupNN.learn(ModeLearn.MINIBATCH, numMiniBatches=8, errorFunct=MEE, verbose=True)
end = time.time()
print("Errore ultima epoch: " + str(errtr[-1]))
print("Tempo: "+str(end-start))

out = open('mlcupresSP_HU70_LR1e-2to1e-4_EP60k_MB8', 'a')
for el in errtr:
    out.write(str(el)+",")
out.write("\n")
out.close()

graphic.grid(True)
graphic.plot(errtr)
graphic.show()
