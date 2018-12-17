"""
Classi Unit e NeuralNetwork che implementano il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
Le classi sono predisposte anche al deep learning, sebbene il learn non lo sia.
"""
from Input import OneOfKAttribute, Input, OneOfKInput, TRInput, OneOfKTRInput 
from ActivFunct import ActivFunct, Sigmoidal, Identity
from DataSet import DataSet, ModeInput
from multiprocessing.dummy import Pool as ThreadPool
import math, time, random
import numpy as np
import matplotlib.pyplot as graphic
from scipy import linalg

from enum import Enum
class ModeLearn(Enum):
    BATCH = 1
    MINIBATCH = 2
    ONLINE = 3

#Superclasse relativa ad una generica unità di rete: non si distingue se l'unità corrente
#sia hidden oppure di output. Il calcolo del delta relativo alla backpropagation (che risulta
# distinto nei due casi sopra citati) si demanda ad opportune sottoclassi.
class Unit(object):

    #pos: posizione dell'unità all'interno del layer
    #dim: numero di archi entranti nell'unità (bias escluso)
    #ValMax: valore massimo del peso di un arco
    #f: funzione di attivazione
    def __init__(self, pos : int, dim : int, ValMax : float, f : ActivFunct, weights: list = None, bias = None, fanIn = 2):

        if weights == None and bias == None:
            if isinstance(f, Sigmoidal):
                rang = 2*ValMax/fanIn

                self.bias = random.uniform(-rang, rang)#peso relativo al bias
                
                weightsList = [random.uniform(-rang,rang) for i in range(dim)]

            else:
                
                self.bias = random.uniform(-ValMax, ValMax)#peso relativo al bias
                
                weightsList = [random.uniform(-ValMax,ValMax) for i in range(dim)]

            self.weights = np.array(weightsList)

        else:
            if len(weights) == dim:
                self.weights = np.array(weights)
                self.bias = bias
            else:
                raise ValueError("weights dim")

        #Memorizzazione della posizione nel layer corrente, del numero di connessioni 
        # al layer precedente.
        self.pos = pos
        self.dim = dim
        self.f = f
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input dell'unità inp.
    def getNet(self, inp : np.array):
        if len(inp) == self.dim:
            
            result = (np.dot(self.weights, inp))
            result += self.bias
            if type(result) == np.int32:
                return int(result)
            else:
                return float(result)
        else:
            raise RuntimeError ("getNet: numbers of weights and inputs don't match.")

    def getWeight(self, index):
        return self.weights[index]

    #Restituisce l'ouput calcolato sull'unità corrente (Net valutato nella funzione di
    # attivazione).
    def getOutput(self, inp: np.array):
        net = self.getNet(inp)
        fval = self.f.getf(net)
        return fval


#Sottoclasse delle unità hidden.
class HiddenUnit(Unit):

    def __init__(self, pos, dim, ValMax, f : ActivFunct, weights: list = None, bias = None, fanIn = 2):
        super().__init__(pos,dim,ValMax,f, weights, bias,fanIn)

    #costruisce il delta da usare nell'algoritmo di backpropagation
    #derivative: derivata prima di f (da vedere se esiste qualche libreria per calcolarla)
    #input: input passato all'unità
    #deltaList: lista dei delta ottenuti al livello soprastante
    #weightsList: lista dei pesi che si riferiscono all'unità 
    def getDelta(self, inp:np.array, deltaOut:np.array, weights_k_j:np.array):

        #Sommatoria(DELTAk * Wkj)
        s = np.dot(deltaOut, weights_k_j)

        net = self.getNet(inp)
        dx = self.f.getDerivative(net)
        
        if type(s) == np.int32:
            s = int(s)
        else:
            s = float(s)

        return s*dx

#Sottoclasse delle unità di output.
class OutputUnit(Unit):

    def __init__(self, pos, dim, ValMax, f : ActivFunct, weights: list = None, bias = None, fanIn = 2):
        super().__init__(pos,dim,ValMax,f, weights, bias, fanIn)
    
    
    #costruisce il delta da usare nell'algoritmo di backpropagation
    #targetOut: valore di target che associato all'input
    #derivative: derivata prima di f (da vedere se esiste qualche libreria per calcolarla)
    #input: input passato all'unità
    def getDelta(self, targetOut, inp:np.array):
        net = self.getNet(inp)
        der = self.f.getDerivative(net)
        err = (targetOut - self.getOutput(inp))
        return err*der
        


#inputLayer: lista di input
#hiddenLayer: lista di hidden units
#outputLayer: lista output units
class NeuralNetwork(object):
    
    #Da usare in fase di testing
    #weights[0]: lista dei pesi delle hidden unit
    #weights[1]: lista dei pesi delle output unit
    def __init__(self, trainingSet: list, f : ActivFunct, new_hyp={}, weights:list = None):
        #Dizionario contenente i settaggi di default (ovviamente modificabili) 
        #degli iperparametri. 
        self.hyp = {'learnRate': 0.1,
                    'momRate':   0.1,
                    'regRate':   0.1,
                    'ValMax':    0.2,
                    'HiddenUnits':  2,
                    'OutputUnits':  1,
                    'MaxEpochs': 10e4,
                    'Tolerance': 10e-4}

        #Aggiornamento degli iperparametri.
        for key in new_hyp:
            if key in self.hyp:
                self.hyp[key] = new_hyp[key]
            else:
                raise ValueError ("new_hyp must be a subdict of hyp!")

        #Lista dei layers, numero degli hidden layers e funzione di attivazione.
        self.hiddenLayer = list()
        self.outputLayer = list()
        self.inputLayer = list()
        self.f = f

        b1 = isinstance(trainingSet[0], TRInput)
        b2 = isinstance(trainingSet[0], OneOfKTRInput)
        b = b1 or b2
        #Inserimento del training set.
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
                self.inputLayer.append(el)
            fanIn = length+1

        #Creazione delle hidden units.
        if weights == None:
            hiddenList = [HiddenUnit(i,length,self.hyp['ValMax'], self.f,fanIn=fanIn) for i in range(self.hyp['HiddenUnits'])]
            self.hiddenLayer = hiddenList
        else:
            if len(weights[0]) == self.hyp["HiddenUnits"]:
                hiddenList = [HiddenUnit(i,length,self.hyp['ValMax'], self.f, weights[0][i],fanIn=fanIn) for i in range(self.hyp['HiddenUnits'])]
                self.hiddenLayer = hiddenList
            else:
                raise ValueError("NN __init__: weights[0] len")
        length = len(hiddenList)

        #Creazione delle output units.
        if weights == None:
            outputList = [OutputUnit(i,length,self.hyp['ValMax'], self.f,fanIn=fanIn) for i in range(self.hyp['OutputUnits'])]
            self.outputLayer = outputList
        else:
            if len(weights[1]) == self.hyp["OutputUnits"]:
                outputList = [OutputUnit(i,length,self.hyp['ValMax'], self.f, weights[1][i],fanIn=fanIn) for i in range(self.hyp['OutputUnits'])]
                self.outputLayer = outputList
            else:
                raise ValueError("NN __init__: weights[1] len")
    
    #Esegue backpropagation e istruisce la rete settando i pesi.
    def learn(self, mode:ModeLearn, errorFunct = None):
        momentum = self.hyp["momRate"]
        learnRate = self.hyp["learnRate"]
        regRate = self.hyp["regRate"]

        oldWeightsRatioOut = np.zeros((len(self.outputLayer), len(self.hiddenLayer)))
        oldWeightsRatioHidden = np.zeros((len(self.hiddenLayer), self.inputLayer[0].getLength()))
        oldratio_Bias_out = np.zeros((len(self.outputLayer), 1))
        oldratio_Bias_hidden = np.zeros((len(self.hiddenLayer), 1))

        epochs = 0
        it = 0
        vecErr = list()
        lErr = np.array([self.getError(self.inputLayer,i,1/(len(self.inputLayer)),errorFunct) for i in range(self.hyp['OutputUnits'])])
        err = linalg.norm(lErr,2)
        vecErr.append(err)
        while(epochs < self.hyp["MaxEpochs"]  and err > self.hyp["Tolerance"]):
            if mode == ModeLearn.BATCH:
                (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden) = self.batchIter(oldWeightsRatioOut, oldWeightsRatioHidden, oldratio_Bias_out, oldratio_Bias_hidden, learnRate, momentum, regRate)
                lErr = np.array([self.getError(self.inputLayer,i,1/(len(self.inputLayer)),errorFunct) for i in range(self.hyp['OutputUnits'])])
                err = linalg.norm(lErr,2)
                vecErr.append(err)
                epochs += 1
            elif mode == ModeLearn.MINIBATCH:
                raise NotImplementedError
            elif mode == ModeLearn.ONLINE:
                if it % len(self.inputLayer) == 0:
                    if it != 0:
                        epochs += 1
                        lErr = np.array([self.getError(self.inputLayer,i,1/(len(self.inputLayer)),errorFunct) for i in range(self.hyp['OutputUnits'])])
                        err = linalg.norm(lErr,2)
                        vecErr.append(err)
                    it = 0
                    arr = list()
                    for el in self.inputLayer:
                        arr.append(el)
                    random.shuffle(arr)
                (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden) = self.onlineIter(oldWeightsRatioOut, oldWeightsRatioHidden, oldratio_Bias_out, oldratio_Bias_hidden, arr.pop(0), learnRate, momentum, regRate)
                it += 1

            
            oldWeightsRatioOut = ratio_W_Out
            oldratio_Bias_out = ratio_Bias_out

            oldWeightsRatioHidden = ratio_W_Hidden
            oldratio_Bias_hidden = ratio_Bias_hidden

        
            """
            if epochs % 50 == 0:
                self.getPlot(vecErr)
            """
        return vecErr
        
    def getPlot(self, val: list):
        array = np.array(val)
        graphic.plot(array)    
        graphic.show()

    
    #Resituisce gli output di rete (array dei valori uscenti dalle unità di output) dato l'input inp.
    def getOutput(self, inp : Input):

        #Calcolo gli outputs delle hidden units
        inp2hiddList = list()
        for u in self.hiddenLayer:
            inp2hiddList.append(u.getOutput(inp.getInput()))

        inp2hidd = np.array(inp2hiddList)

        #calcolo gli output di rete
        hidd2outList = list()
        for u in self.outputLayer:
            hidd2outList.append(u.getOutput(inp2hidd))

        return np.array(hidd2outList)
        
        
    #Calcola l'errore (rischio) empirico della lista di TRInput o OneOfKTRInput data, sulla i-esima
    #unità di output (Att: i va da 0 ad OutputUnits-1!) con la funzione L (loss)
    # eventualmente assegnata, default=LMS e fattore di riscalamento k.
    def getError(self, data : list, i : int, k, L= None):
        if L == None:
            L = lambda target,value: (target - value)**2
        #Controllo di validità dei dati.
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

        #Controllo di validità dell'indice i.
        if not i in range(self.hyp['OutputUnits']):
            raise RuntimeError ("Index i out of bounds")
        
        #Calcolo effettivo dell'errore.
        s = 0
        for d in data:
            if isinstance(self.f, Sigmoidal):
                target = d.getTargetSigmoidal()
            else:
                target = d.getTarget()
            s += L(target,self.getOutput(d)[i])
        return k*s

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

        ratio_W_Out = np.zeros((len(self.outputLayer), len(self.hiddenLayer)))
        ratio_W_Hidden = np.zeros((len(self.hiddenLayer), self.inputLayer[0].getLength()))
        ratio_Bias_out = np.zeros((len(self.outputLayer), 1))
        ratio_Bias_hidden = np.zeros((len(self.hiddenLayer), 1))

        #scorro sugli input
        for inp in self.inputLayer:

            #calcolo gli output forniti dalle unità hidden
            hiddenOutResultsList = list()
            for unit in self.hiddenLayer:
                hiddenOutResultsList.append(unit.getOutput(inp.getInput()))

            hiddenOutResults = np.array(hiddenOutResultsList)
            
            #calcolo i delta relativi alle unità di output
            deltaOutResultsList = list()
            for unit in self.outputLayer:
                deltaOutResultsList.append(unit.getDelta(inp.getTarget(), hiddenOutResults))

            deltaOutResults = np.array(deltaOutResultsList)

            #calcolo i delta relativi alle unità di hidden
            deltaHiddenResultsList = list()
            for unit in self.hiddenLayer:
                wList = list()
                for out in self.outputLayer:
                    wList.append(out.getWeight(unit.pos))
                deltaHiddenResultsList.append(unit.getDelta(inp.getInput(), deltaOutResults, np.array(wList)))

            deltaHiddenResults = np.array(deltaHiddenResultsList)

            for hUnit in self.hiddenLayer:
                t = hUnit.pos
                ratio_W_Hidden[t] += learnRate*deltaHiddenResults[t]*inp.getInput()
                ratio_Bias_hidden[t] += learnRate*deltaHiddenResults[t]

            for oUnit in self.outputLayer:
                t = oUnit.pos
                ratio_W_Out[t] += learnRate*deltaOutResults[t]*hiddenOutResults
                ratio_Bias_out[t] += learnRate*deltaOutResults[t]

            
        ratio_W_Out /= len(self.inputLayer)
        ratio_W_Hidden /= len(self.inputLayer)
        ratio_Bias_out /= len(self.inputLayer)
        ratio_Bias_hidden /= len(self.inputLayer)

        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_W_Hidden += momRate * oldWeightsRatioHidden
        ratio_Bias_out += momRate * oldBiasRatioOut
        ratio_Bias_hidden += momRate * oldBiasRatioHidden

        #aggiornamento pesi unità di output
        for oUnit in self.outputLayer:
            t = oUnit.pos
            oUnit.weights = oUnit.weights + ratio_W_Out[t] - regRate*oUnit.weights
            oUnit.bias = oUnit.bias + ratio_Bias_out[t]

        #aggiornamento pesi unità hidden layer
        for hUnit in self.hiddenLayer:
            t = hUnit.pos
            hUnit.weights = hUnit.weights + ratio_W_Hidden[t] - regRate*hUnit.weights
            hUnit.bias = hUnit.bias + ratio_Bias_hidden[t]

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

        ratio_W_Out = np.zeros((len(self.outputLayer), len(self.hiddenLayer)))
        ratio_W_Hidden = np.zeros((len(self.hiddenLayer), self.inputLayer[0].getLength()))
        ratio_Bias_out = np.zeros((len(self.outputLayer), 1))
        ratio_Bias_hidden = np.zeros((len(self.hiddenLayer), 1))
        
        #calcolo gli output forniti dalle unità hidden
        hiddenOutResultsList = list()
        for unit in self.hiddenLayer:
            hiddenOutResultsList.append(unit.getOutput(inp.getInput()))

        hiddenOutResults = np.array(hiddenOutResultsList)
        
        #calcolo i delta relativi alle unità di output
        deltaOutResultsList = list()
        for unit in self.outputLayer:
            deltaOutResultsList.append(unit.getDelta(inp.getTarget(), hiddenOutResults))

        deltaOutResults = np.array(deltaOutResultsList)

        #calcolo i delta relativi alle unità di input
        deltaHiddenResultsList = list()
        for unit in self.hiddenLayer:
            wList = list()
            for out in self.outputLayer:
                wList.append(out.getWeight(unit.pos))
            deltaHiddenResultsList.append(unit.getDelta(inp.getInput(), deltaOutResults, np.array(wList)))

        deltaHiddenResults = np.array(deltaHiddenResultsList)

        for hUnit in self.hiddenLayer:
            t = hUnit.pos
            ratio_W_Hidden[t] += learnRate*deltaHiddenResults[t]*inp.getInput()
            ratio_Bias_hidden[t] += learnRate*deltaHiddenResults[t]

        for oUnit in self.outputLayer:
            t = oUnit.pos
            ratio_W_Out[t] += learnRate*deltaOutResults[t]*hiddenOutResults
            ratio_Bias_out[t] += learnRate*deltaOutResults[t]

        ratio_W_Out = (1 - momRate)*ratio_W_Out + momRate * oldWeightsRatioOut
        ratio_W_Hidden = (1 - momRate)*ratio_W_Hidden + momRate * oldWeightsRatioHidden
        ratio_Bias_out = (1-momRate)*ratio_Bias_out + momRate * oldBiasRatioOut
        ratio_Bias_hidden = (1-momRate)*ratio_Bias_hidden + momRate * oldBiasRatioHidden

        #aggiornamento pesi unità di output
        for oUnit in self.outputLayer:
            t = oUnit.pos
            oUnit.weights = oUnit.weights + ratio_W_Out[t] - regRate*oUnit.weights
            oUnit.bias = oUnit.bias + ratio_Bias_out[t]

        #aggiornamento pesi unità hidden layer
        for hUnit in self.hiddenLayer:
            t = hUnit.pos
            hUnit.weights = hUnit.weights + ratio_W_Hidden[t] - regRate*hUnit.weights
            hUnit.bias = hUnit.bias + ratio_Bias_hidden[t]

        return (ratio_W_Out, ratio_W_Hidden, ratio_Bias_out, ratio_Bias_hidden)
    
#Test.
f = Sigmoidal(8)

domains = [3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = 1

trainingSet = DataSet("monks-2.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
testSet = DataSet("monks-2.test", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)

neruale = NeuralNetwork(trainingSet.inputList, f, {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.005, 'MaxEpochs': 6000})
errl = neruale.learn(ModeLearn.BATCH)
neruale.getPlot(errl)

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

s = 0
for d in testSet.inputList:
    out = (neruale.getOutput(d)[0] >= 0.5)
    s += abs(out - d.getTarget())
perc = 1 - s/len(testSet.inputList)
print("Accuratezza sul test set: " + str(perc*100) + "%.")


