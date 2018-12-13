"""
Classi Unit e NeuralNetwork che implementano il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
Le classi sono predisposte anche al deep learning, sebbene il learn non lo sia.
"""
from Input import Attribute, OneOfKAttribute, Input, OneOfKInput, TRInput, OneOfKTRInput 
from ActivFunct import ModeActiv, ActivFunct
from DataSet import DataSet, ModeInput
from multiprocessing.dummy import Pool as ThreadPool
import scipy, math, time
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
    def __init__(self, pos : int, dim : int, ValMax : float, f : ActivFunct, weights: list = None, fanIn = 2):

        if weights == None:
            from random import uniform

            rang = 2*ValMax/fanIn
            self.weights = [uniform(-rang,rang) for i in range(dim + 1)]

        else:
            if len(weights) == dim+1:
                self.weights = weights
            else:
                raise ValueError("weights dim")

        #Memorizzazione della posizione nel layer corrente, del numero di connessioni 
        # al layer precedente.
        self.pos = pos
        self.dim = dim
        self.f = f
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input dell'unità inp.
    def getNet(self, inp : list):
        if len(inp) == self.dim:
            #Primo passo: bias=1 * self.weights[0]
            s = self.weights[0]

            for i in range(len(inp)):
                el = inp[i]
                if not isinstance(el,(int,float)):
                    raise RuntimeError ("getNet: passed non-number input element.")
                
                s += self.weights[i+1]*el
            return s
        else:
            raise RuntimeError ("getNet: numbers of weights and inputs don't match.")

    def getWeight(self, index):
        return self.weights[index]

    #Restituisce l'ouput calcolato sull'unità corrente (Net valutato nella funzione di
    # attivazione).
    def getOutput(self, inp: list):
        net = self.getNet(inp)
        fval = self.f.getf(net)
        return fval


#Sottoclasse delle unità hidden.
class HiddenUnit(Unit):

    def __init__(self, pos, dim, ValMax, f : ActivFunct, weights: list = None, fanIn = 2):
        super().__init__(pos,dim,ValMax,f, weights,fanIn)

    #costruisce il delta da usare nell'algoritmo di backpropagation
    #derivative: derivata prima di f (da vedere se esiste qualche libreria per calcolarla)
    #input: input passato all'unità
    #deltaList: lista dei delta ottenuti al livello soprastante
    #weightsList: lista dei pesi che si riferiscono all'unità 
    def getDelta(self, input:list, deltaList:list, weightsList:list):
        s = 0

        #Sommatoria(DELTAk * Wkj)
        for i in range(0, len(deltaList)):
            s += deltaList[i]*weightsList[i]

        net = self.getNet(input)
        dx = self.f.getDerivative(net)
        return s*dx

#Sottoclasse delle unità di output.
class OutputUnit(Unit):

    def __init__(self, pos, dim, ValMax, f : ActivFunct, weights: list = None, fanIn = 2):
        super().__init__(pos,dim,ValMax,f, weights, fanIn)
    
    
    #costruisce il delta da usare nell'algoritmo di backpropagation
    #targetOut: valore di target che associato all'input
    #derivative: derivata prima di f (da vedere se esiste qualche libreria per calcolarla)
    #input: input passato all'unità
    def getDelta(self, targetOut, input:list):
        return (targetOut - self.getOutput(input))*self.f.getDerivative(self.getNet(input))
        


#layers[0]: inputs
#layers[1]: hidden units
#layers[2]: output units
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
                    'HiddenLayers': 1,
                    'HiddenUnits':  2,
                    'OutputUnits':  1,
                    'MaxIter': 10e4,
                    'Tolerance': 10e-4}

        #Aggiornamento degli iperparametri.
        for key in new_hyp:
            if key in self.hyp:
                self.hyp[key] = new_hyp[key]
            else:
                raise ValueError ("new_hyp must be a subdict of hyp!")

        #Lista dei layers, numero degli hidden layers e funzione di attivazione.
        self.layers = []
        self.noHiddLayers = self.hyp['HiddenLayers']
        self.f = f

        #Inserimento del training set.
        if len(trainingSet) == 0 or not isinstance(trainingSet[0], TRInput and OneOfKTRInput):
            raise ValueError ("inserted TR set is not valid!")
        else:
            length = trainingSet[0].getLength()
            for el in trainingSet:
                if not isinstance(el, TRInput and OneOfKTRInput) or el.getLength() != length:
                    raise ValueError ("TR set not valid: not homogeneous")
            self.layers.append(trainingSet.copy())
            fanIn = self.layers[0][0].len()+1

        #Creazione delle hidden units.
        for j in range(self.noHiddLayers):
            if weights == None:
                hiddenList = [HiddenUnit(i,length,self.hyp['ValMax'], self.f,fanIn=fanIn) for i in range(self.hyp['HiddenUnits'])]
                self.layers.append(hiddenList)
            else:
                if len(weights[0]) == self.hyp["HiddenUnits"]:
                    hiddenList = [HiddenUnit(i,length,self.hyp['ValMax'], self.f, weights[0][i],fanIn=fanIn) for i in range(self.hyp['HiddenUnits'])]
                    self.layers.append(hiddenList)
                else:
                    raise ValueError("NN __init__: weights[0] len")
            length = len(hiddenList)

        #Creazione delle output units.
        if weights == None:
            outputList = [OutputUnit(i,length,self.hyp['ValMax'], self.f,fanIn=fanIn) for i in range(self.hyp['OutputUnits'])]
            self.layers.append(outputList)
        else:
            if len(weights[1]) == self.hyp["OutputUnits"]:
                outputList = [OutputUnit(i,length,self.hyp['ValMax'], self.f, weights[1][i],fanIn=fanIn) for i in range(self.hyp['OutputUnits'])]
                self.layers.append(outputList)
            else:
                raise ValueError("NN __init__: weights[1] len")

    #Esegue backpropagation e istruisce la rete settando i pesi.
    def learn(self, mode:ModeLearn, errorFunct = None):
        if self.noHiddLayers == 1:
            momentum = self.hyp["momRate"]
            learnRate = self.hyp["learnRate"]
            regRate = self.hyp["regRate"]
            oldWeightsRatioOut = scipy.zeros((len(self.layers[2]), len(self.layers[1])+ 1))
            oldWeightsRatioHidden = scipy.zeros((len(self.layers[1]), self.layers[0][0].len() + 1))
            it = 0
            vecErr = []
            lErr = scipy.array([self.getError(self.layers[0],i,1/(len(self.layers[0])),errorFunct) for i in range(self.hyp['OutputUnits'])])
            err = linalg.norm(lErr,2)
            vecErr.append(err)
            while(it < self.hyp["MaxIter"]  and err > self.hyp["Tolerance"]):
                if mode == ModeLearn.BATCH:
                    t1 = time.time_ns()
                    self.batchIter(oldWeightsRatioOut, oldWeightsRatioHidden, learnRate, momentum, regRate)
                    t2 = time.time_ns()
                    res = t2 - t1
                elif mode == ModeLearn.MINIBATCH:
                    raise NotImplementedError
                elif mode == ModeLearn.ONLINE:
                    raise NotImplementedError

                lErr = scipy.array([self.getError(self.layers[0],i,1/(len(self.layers[0])),errorFunct) for i in range(self.hyp['OutputUnits'])])
                err = linalg.norm(lErr,2)
                vecErr.append(err)
                it += 1
                if it % 100 == 0:
                    graphic.plot(scipy.array(vecErr),'--')
                    graphic.show()
                    print(err)
        else:
            raise NotImplementedError ("Deep learning models not already implemented.")
    
    #Restituisce copia della lista rappresentante il layer i-esimo.
    def getLayer(self,i):
        if i in range(len(self.layers)):
            return self.layers[i].copy()
        else:
            raise RuntimeError ("Index i out of bounds.")
    
    #Resituisce la lista degli output di rete (lista dei valori uscenti dalle unità di output) dato l'input inp.
    def getOutput(self, inp : Input):
        #Da inp costruisco la corrispondente lista.
        valList = list()
        valList = inp.getInput()

        #Calcolo gli outputs delle unità sui layers successivi.
        for i in range(1,len(self.layers)):
            valList = [unit.getOutput(valList) for unit in self.getLayer(i)]
        return valList
        
        
    #Calcola l'errore (rischio) empirico della lista di TRInput o OneOfKTRInput data, sulla i-esima
    #unità di output (Att: i va da 0 ad OutputUnits-1!) con la funzione L (loss)
    # eventualmente assegnata, default=LMS e fattore di riscalamento k.
    def getError(self, data : list, i : int, k, L= None):
        if L == None:
            L = lambda target,value: (target - value)**2
        #Controllo di validità dei dati.
        if len(data) == 0 or not isinstance(data[0], TRInput and OneOfKTRInput):
            raise ValueError ("inserted set is not valid!")
        else:
            length = data[0].getLength()
            for el in data:
                if not isinstance(el, TRInput and OneOfKTRInput) or el.getLength() != length:
                    raise ValueError ("data set not valid: not homogeneous")

        #Controllo di validità dell'indice i.
        if not i in range(self.hyp['OutputUnits']):
            raise RuntimeError ("Index i out of bounds")
        
        #Calcolo effettivo dell'errore.
        s = 0
        for d in data:
            s += L(d.getTarget(),self.getOutput(d)[i])
        return k*s

    #region Thread Deltas utils
    def getHiddenOut(self, unit: HiddenUnit, inp):
        return unit.getOutput(inp)
        
    def getDeltaOut(self, unit: OutputUnit, inp, target):
        return unit.getDelta(target, inp)

    def getDeltaHidden(self, unit: HiddenUnit, input, deltaList, outUnits: list):
        weights = list()
        hiddenUnitIndex = unit.pos
        for outUnit in outUnits:
            if isinstance(outUnit, OutputUnit):
                weights.append(outUnit.getWeight(hiddenUnitIndex+1))
            else:
                raise ValueError("in getDeltaHidden, outUnits type error")   
        return unit.getDelta(input, deltaList, weights)

    #endregion

    """
    Questo metodo permette di calcolare i delta della backpropagation in parallelo

    -param
        inp: input per il quale calcolare i delta
    -result
        Tupla<Lista dei delta relativi alle unità di output,  Lista dei delta relativi alle unità hidden>
    """
    def getDeltas(self, inp: Input, nThread: int = -1):
        import functools

        if not isinstance(inp, TRInput and OneOfKTRInput):
            raise ValueError ("inserted input is not valid!")

        if nThread == 0:
            hiddenOutResults = list()
            for unit in self.layers[1]:
                hiddenOutResults.append(unit.getOutput(inp.getInput()))

            deltaOutResults = list()
            for unit in self.layers[2]:
                deltaOutResults.append(unit.getDelta(inp.getTarget(), hiddenOutResults))

            deltaHiddenResults = list()

            for unit in self.layers[1]:
                wList = list()
                for out in self.layers[2]:
                    wList.append(out.getWeight(unit.pos + 1))
                deltaHiddenResults.append(unit.getDelta(inp.getInput(), deltaOutResults, wList))
            result = (deltaOutResults, deltaHiddenResults)
            return result
        else:
            if nThread < 0:
            #calcolo ouput delle unità hidden, che costituiscono l'input per le unità output
                pool = ThreadPool()
            else:
                pool = ThreadPool(nThread)

            hiddenOutResults = pool.map(functools.partial(self.getHiddenOut, inp=inp.getInput()), self.layers[1])
            pool.close()
            pool.join()
            
            #calcolo dei delta delle unità di output
            pool = ThreadPool()
            targetOut = inp.getTarget()
            deltaOutResults = pool.map(functools.partial(self.getDeltaOut, target=targetOut,inp=hiddenOutResults), self.layers[2])
            pool.close()
            pool.join()

            #calcolo dei delta delle unità hidden
            pool = ThreadPool()
            deltaHiddenResults = pool.map(functools.partial(self.getDeltaHidden, input=inp.getInput(), deltaList=deltaOutResults, outUnits=self.layers[2]), self.layers[1])
            pool.close()
            pool.join()
            result = (deltaOutResults, deltaHiddenResults)
            return result


    def updateRatio(self, layer: int, ratio_W: scipy.array, delta: list, inp: Input, learnRate):
        if layer > 2 or layer < 1:
            raise ValueError("in updateRatio 1 <= layer <= 2")

        #itero sulle unità del livello corrente
        for unit in self.layers[layer]:

            #itero sui pesi della unità selezionata
            for weight in unit.weights:
                i = unit.pos
                j = unit.weights.index(weight)

                #Memorizzo l'ouput dell'unità j del layer precedente sulla variabile outj,
                #generalizzando i casi input, bias ed hidden.
                if j == 0:
                    #Bias
                    outj = 1  
                elif layer == 2:
                    #Hidden: attenzione che il weight j=0 corrisponde al bias, devo
                    #dunque accedere all'unità in posizione j-1 corrispondentemente ad
                    #ogni peso j non zero.
                    outj = self.layers[layer-1][j-1].getOutput(inp.getInput()) 
                else:
                    #Input
                    outj = inp.getInput()[j-1]

                ratio_W[i,j] += learnRate * delta[i] * outj

    """
    Metodo che implementa una iterazione dell'algoritmo batch backprop

    -arguments
        oldWeightsRatio: lista delle variazioni dei pesi all'iterazione precedente
    """
    def batchIter(self, oldWeightsRatioOut: scipy.array, oldWeightsRatioHidden: scipy.array, learnRate, momRate, regRate):

        ratio_W_Out = scipy.zeros((len(self.layers[2]), len(self.layers[1]) + 1))
        #Errore: in self.layers[0] ho salvato una lista di input, a me serve invece il numero
        #di attributi del singolo input, prendo l'input in posizione 0 come riferimento visto 
        #che ho già fatto il controllo di omogeneità in fase di inizializzazione.
        ratio_W_Hidden = scipy.zeros((len(self.layers[1]), self.layers[0][0].len() + 1))

        #scorro sugli input
        for inp in self.layers[0]:
            (outDelta, hiddenDelta) = self.getDeltas(inp, 0)

            self.updateRatio(2, ratio_W_Out, outDelta, inp, learnRate)
            self.updateRatio(1, ratio_W_Hidden, hiddenDelta, inp, learnRate)

        ratio_W_Out /= len(self.layers[0])
        ratio_W_Hidden /= len(self.layers[0])

        ratio_W_Out += momRate * oldWeightsRatioOut
        ratio_W_Hidden += momRate * oldWeightsRatioHidden

        oldWeightsRatioOut = ratio_W_Out
        oldWeightsRatioHidden = ratio_W_Hidden

        for i in range (len(self.layers[2])):
            for j in range (len(self.layers[1])+1):
                w_i_j = self.layers[2][i].weights[j]
                self.layers[2][i].weights[j] += ratio_W_Out[i,j] - regRate * w_i_j

        for i in range (len(self.layers[1])):
            for j in range (self.layers[0][0].len()+1):
                w_i_j = self.layers[1][i].weights[j]
                self.layers[1][i].weights[j] += ratio_W_Hidden[i,j] - regRate * w_i_j

    
        
#Test.
from math import exp
a1=OneOfKAttribute(5,3)
a2=OneOfKAttribute(4,2)
i1=OneOfKTRInput([a1,a2],False)

a1=OneOfKAttribute(5,1)
a2=OneOfKAttribute(4,3)
a3=OneOfKAttribute(2,1)
i2=OneOfKTRInput([a1,a2],True)
i3=OneOfKInput([a1,a2,a3])

f = ActivFunct(param=[10])

il=[i1,i2]
n = NeuralNetwork(il,f)
out = n.getOutput(i2)
print(out)
print(n.getError([i2],0,1))

l = i2.getInput()
linp = list()
for el in l:
    linp.append(int(el))
print("\n\n\n\nsigmoid")
print("input: "+str(linp)) 

outUnit = OutputUnit(1, 1, 0.2, f)
hiddenUnit = HiddenUnit(1, len(linp), 0.2, f)

hout = list()
hout.append(hiddenUnit.getOutput(linp))
onet = outUnit.getNet(hout)
outputOut = outUnit.getOutput(hout)
outputDelta = outUnit.getDelta(1, hout)

print("output delta:" +str(outputDelta))
hnet = hiddenUnit.getNet(linp)
hiddenDelta = hiddenUnit.getDelta(linp, [outputDelta], [1])
print("hidden delta:" +str(hiddenDelta))

result = n.getDeltas(i1)

print (result)

weights = list()
a = OneOfKAttribute(1, 1)
i1 = OneOfKTRInput([a], True)
il = [i1]
weights.append(list())
l = list()
for i in range(0, 2):
    l.append(1)

weights[0] = [l, l]

l2 = [1, 1, 1]
weights.append(list())
weights[1] = [l2]

nn2 = NeuralNetwork(il, f, weights= weights)
print(nn2.getOutput(i1))


domains = [3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = 1
trainingSet = DataSet("monks-3.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
testSet = DataSet("monks-3.test", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)

neruale = NeuralNetwork(trainingSet.inputList, f, {'HiddenUnits':4, 'learnRate':0.1, 'ValMax':0.7, 'momRate':0.6, 'regRate':0, 'Tolerance':0.006})
neruale.learn(ModeLearn.BATCH)

s = 0
for d in testSet.inputList:
    out = (neruale.getOutput(d)[0] >= 0.5)
    s += abs(out - d.getTargetExact())
perc = 1 - s/len(testSet.inputList)
print("Accuratezza sul test set: " + str(perc*100) + "%.")
#ValueError: Inserted input is not valid for this NN!
#n.getOutput(i3)
