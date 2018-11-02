"""
Classi Unit e NeuralNetwork che implementano il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
"""
from Input import Attribute, Input, TRInput

#Superclasse relativa ad una generica unità di rete: non si distingue se l'unità corrente
#sia hidden oppure di output. Il calcolo del delta relativo alla backpropagation (che risulta
# distinto nei due casi sopra citati) si demanda ad opportune sottoclassi.
class Unit(object):

    def __init__(self, pos : int, dim : int, ValMax : float, network : object, f):
        from random import uniform
        #Inizializzazione dei pesi a valori casuali (distr. uniforme).
        self.weights = [uniform(0,ValMax) for i in range(dim)]

        #Memorizzazione del layer corrente, del numero di connessioni al layer precedente,
        #della rete neurale relativa all'unità, della funzione di attivazione.
        self.pos = pos
        self.dim = dim
        self.net = network
        self.activation = f
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input di rete inp.
    def getNet(self, inp: Input):
        s = 0
        if self.pos == 1:
            for i in range(inp.getLength()):
                i = self.net.getLayer(0).index(inp)
                s += self.weights[i]*inp.getValue(i)
        else:
            for unit in self.net.getLayer(self.pos-1):
                i = self.net.getLayer(self.pos-1).index(unit)
                s += self.weights[i]*unit.getOutput(inp)
        return s

    #Restituisce l'ouput calcolato sull'unità corrente (Net valutato nella funzione di
    # attivazione).
    def getOutput(self, inp: Input):
        return(self.activation(self.getNet(inp)))

#Sottoclasse delle unità hidden.
class HiddenUnit(Unit):

    def __init__(self, pos, dim, ValMax, network, f):
        super().__init__(pos,dim,ValMax,network,f)
    
    def getDelta(self):
        raise NotImplementedError

#Sottoclasse delle unità di output.
class OutputUnit(Unit):

    def __init__(self, pos, dim, ValMax, network, f):
        super().__init__(pos,dim,ValMax,network,f)
    
    def getDelta(self):
        raise NotImplementedError


class NeuralNetwork(object):
    
    def __init__(self, trainingSet: list,  new_hyp={}):
        from math import exp
        #Dizionario contenente i settaggi di default (ovviamente modificabili) 
        #degli iperparametri. 
        self.hyp = {'eta':         0.1,
                    'alpha':       0.1,
                    'lambda':      0.1,
                    'ValMax':      0.2,
                    'HiddenUnits': 2,
                    'OutputUnits': 1, 
                    'ActivFun': lambda x: 1/(1+(exp(-x)))}

        #Aggiornamento degli iperparametri.
        for key in new_hyp:
            if key in self.hyp:
                self.hyp[key] = new_hyp[key]
            else:
                raise ValueError ("new_hyp must be a subdict of hyp!")

        #Lista dei layers.
        self.layers = []

        #Inserimento in input del training set.
        if len(trainingSet) == 0 or not isinstance(trainingSet[0], TRInput):
            raise ValueError ("inserted TR set is not valid!")
        else:
            length = trainingSet[0].getLength()
            for el in trainingSet:
                if not isinstance(el, TRInput) or el.getLength() != length:
                    raise ValueError ("TR set not valid: not homogeneous")
            self.layers.append(trainingSet.copy())

        #Creazione delle hidden units.
        hiddenList = []
        for i in range(self.hyp['HiddenUnits']):
                hiddenList.append(HiddenUnit(1,length,self.hyp['ValMax'],self,self.hyp['ActivFun']))
        self.layers.append(hiddenList)

        #Creazione delle output units.
        outputList = []
        for i in range(self.hyp['OutputUnits']):
                outputList.append(OutputUnit(2,len(hiddenList),self.hyp['ValMax'],self,self.hyp['ActivFun']))
        self.layers.append(outputList)

    #Esegue backpropagation e istruisce la rete settando i pesi.
    def learn(self):
        raise NotImplementedError
    
    #Restituisce copia della lista rappresentante il layer i-esimo.
    def getLayer(self,i):
        if i in range(len(self.layers)):
            return self.layers[i].copy()
        else:
            raise RuntimeError ("Index i out of bounds.")

    #Restituisce la lista degli output di rete dato un input.
    def getOutput(self, inp : Input):
        out_list = []
        for outnode in self.layers[2]:
            out_list.append(outnode.getOutput(inp))
        return out_list
    
    #Calcola l'errore (rischio) empirico della lista di TRInput data, sulla i-esima
    #unità di output (Att: i va da 0 ad OutputUnits-1!) con la regola dei LS riscalata
    #per il fattore k. Ad es. k=1/len(data) da' LMS.
    def getError(self, data : list, i : int, k : int):
        #Controllo di validità dei dati.
        if len(data) == 0 or not isinstance(data[0], TRInput):
            raise ValueError ("inserted set is not valid!")
        else:
            length = data[0].getLength()
            for el in data:
                if not isinstance(el, TRInput) or el.getLength() != length:
                    raise ValueError ("data set not valid: not homogeneous")

        #Controllo di validità dell'indice i.
        if not i in range(self.hyp['OutputUnits']):
            raise RuntimeError ("Index i out of bounds")
        
        #Calcolo effettivo dell'errore.
        s = 0
        for d in data:
            s += (d.getTarget() - self.getOutput(d)[i])**2
        return k*s


#Test.
a1=Attribute(5,3)
a2=Attribute(4,2)
i1=TRInput([a1,a2],False)

a1=Attribute(5,1)
a2=Attribute(4,3)
i2=TRInput([a1,a2],True)

il=[i1,i2]
n = NeuralNetwork(il)
print(n.getOutput(i1))
print(n.getError([i2],0,1))

