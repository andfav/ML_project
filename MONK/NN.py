"""
Classi Unit e NeuralNetwork che implementano il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
Le classi sono predisposte anche al deep learning, sebbene il learn non lo sia.
"""
from Input import Attribute, OneOfKAttribute, Input, OneOfKInput, TRInput, OneOfKTRInput 
from ActivFunct import ActivFunct

#Superclasse relativa ad una generica unità di rete: non si distingue se l'unità corrente
#sia hidden oppure di output. Il calcolo del delta relativo alla backpropagation (che risulta
# distinto nei due casi sopra citati) si demanda ad opportune sottoclassi.
class Unit(object):

    #pos: posizione dell'unità all'interno del layer
    #dim: numero di archi entranti nell'unità
    #ValMax: valore massimo del peso di un arco
    #f: funzione di attivazione
    def __init__(self, pos : int, dim : int, ValMax : float, f : ActivFunct):
        from random import uniform
        #Inizializzazione dei pesi a valori casuali (distr. uniforme).
        self.weights = [uniform(0,ValMax) for i in range(dim)]

        #Memorizzazione della posizione nel layer corrente, del numero di connessioni 
        # al layer precedente.
        self.pos = pos
        self.dim = dim
        self.f = f
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input dell'unità inp.
    def getNet(self, inp : list):
        if len(inp) == self.dim:
            s = 0
            for i in range(len(inp)):
                el = inp[i]
                if not isinstance(el,(int,float)):
                    raise RuntimeError ("getNet: passed non-number input element.")
                #i = inp.index(el) errore perchè se 2 elem hanno lo stesso valore non funziona

                s += self.weights[i]*el
            return s
        else:
            raise RuntimeError ("getNet: numbers of weights and inputs don't match.")

    #Restituisce l'ouput calcolato sull'unità corrente (Net valutato nella funzione di
    # attivazione).
    def getOutput(self, inp: list):
        net = self.getNet(inp)
        fval = self.f.getf(net)
        return fval

#Sottoclasse delle unità hidden.
class HiddenUnit(Unit):

    def __init__(self, pos, dim, ValMax, f : ActivFunct):
        super().__init__(pos,dim,ValMax,f)

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

    def __init__(self, pos, dim, ValMax, f : ActivFunct):
        super().__init__(pos,dim,ValMax,f)
    
    #costruisce il delta da usare nell'algoritmo di backpropagation
    #targetOut: valore di target che associato all'input
    #derivative: derivata prima di f (da vedere se esiste qualche libreria per calcolarla)
    #input: input passato all'unità
    def getDelta(self, targetOut, input:list):
        return (targetOut - self.getOutput(input))*self.f.getDerivative(self.getNet(input))
        



class NeuralNetwork(object):
    
    def __init__(self, trainingSet: list, f : ActivFunct, new_hyp={}):
        from math import exp
        #Dizionario contenente i settaggi di default (ovviamente modificabili) 
        #degli iperparametri. 
        self.hyp = {'eta':         0.1,
                    'alpha':       0.1,
                    'lambda':      0.1,
                    'ValMax':      0.2,
                    'HiddenLayers': 1,
                    'HiddenUnits':  2,
                    'OutputUnits':  1}

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

        #Creazione delle hidden units.
        for j in range(self.noHiddLayers):
            hiddenList = [HiddenUnit(j+1,length,self.hyp['ValMax'], self.f) for i in range(self.hyp['HiddenUnits'])]
            self.layers.append(hiddenList)
            length = len(hiddenList)

        #Creazione delle output units.
        outputList = [OutputUnit(self.noHiddLayers+1,length,self.hyp['ValMax'], self.f) for i in range(self.hyp['OutputUnits'])]
        self.layers.append(outputList)

    #Esegue backpropagation e istruisce la rete settando i pesi.
    def learn(self):
        if self.noHiddLayers == 1:
            raise NotImplementedError
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
    def getError(self, data : list, i : int, k, L=lambda target,value: (target - value)**2):
        
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

#prova con sigmoid
def sigmoid(a,x): 
    return 1/(1+(exp(-a*x)))

f = ActivFunct(sigmoid,[1])

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
#


#ValueError: Inserted input is not valid for this NN!
#n.getOutput(i3)

