"""
Classi Unit e NeuralNetwork che implementano il nostro modello di rete neurale,
in particolare è utilizzato un algoritmo di apprendimento backpropagation di tipo
batch.
Le classi sono predisposte anche al deep learning, sebbene il learn non lo sia.
"""
from Input import Attribute, Input, TRInput

#Superclasse relativa ad una generica unità di rete: non si distingue se l'unità corrente
#sia hidden oppure di output. Il calcolo del delta relativo alla backpropagation (che risulta
# distinto nei due casi sopra citati) si demanda ad opportune sottoclassi.
class Unit(object):

    def __init__(self, pos : int, dim : int, ValMax : float):
        from random import uniform
        #Inizializzazione dei pesi a valori casuali (distr. uniforme).
        self.weights = [uniform(0,ValMax) for i in range(dim)]

        #Memorizzazione della posizione nel layer corrente, del numero di connessioni 
        # al layer precedente.
        self.pos = pos
        self.dim = dim
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input dell'unità inp.
    def getNet(self, inp : list):
        if len(inp) == self.dim:
            s = 0
            for el in inp:
                if not isinstance(el,(int,float)):
                    raise RuntimeError ("getNet: passed non-number input element.")
                i = inp.index(el)
                s += self.weights[i]*el
            return s
        else:
            raise RuntimeError ("getNet: numbers of weights and inputs don't match.")

    #Restituisce l'ouput calcolato sull'unità corrente (Net valutato nella funzione di
    # attivazione).
    def getOutput(self, inp: list, f):
        return(f(self.getNet(inp)))

#Sottoclasse delle unità hidden.
class HiddenUnit(Unit):

    def __init__(self, pos, dim, ValMax):
        super().__init__(pos,dim,ValMax)

    def getDelta(self):
        raise NotImplementedError

#Sottoclasse delle unità di output.
class OutputUnit(Unit):

    def __init__(self, pos, dim, ValMax):
        super().__init__(pos,dim,ValMax)
    
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
                    'HiddenLayers': 1,
                    'HiddenUnits':  2,
                    'OutputUnits':  15, 
                    'ActivFun': lambda x: 1/(1+(exp(-x)))}

        #Aggiornamento degli iperparametri.
        for key in new_hyp:
            if key in self.hyp:
                self.hyp[key] = new_hyp[key]
            else:
                raise ValueError ("new_hyp must be a subdict of hyp!")

        #Lista dei layers e numero degli hidden layers.
        self.layers = []
        self.noHiddLayers = self.hyp['HiddenLayers']

        #Inserimento del training set.
        if len(trainingSet) == 0 or not isinstance(trainingSet[0], TRInput):
            raise ValueError ("inserted TR set is not valid!")
        else:
            length = trainingSet[0].getLength()
            for el in trainingSet:
                if not isinstance(el, TRInput) or el.getLength() != length:
                    raise ValueError ("TR set not valid: not homogeneous")
            self.layers.append(trainingSet.copy())

        #Creazione delle hidden units.
        for j in range(self.noHiddLayers):
            hiddenList = [HiddenUnit(j+1,length,self.hyp['ValMax']) for i in range(self.hyp['HiddenUnits'])]
            self.layers.append(hiddenList)
            length = len(hiddenList)

        #Creazione delle output units.
        outputList = [OutputUnit(self.noHiddLayers+1,length,self.hyp['ValMax']) for i in range(self.hyp['OutputUnits'])]
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

    #Restituisce la lista degli output al layer l, dato l'input inp e la 
    #funzione f.
    #Nota: usa f = lambda x : x per la lista dei net al layer l.
    def getLayerOutput(self, inp : Input, l : int, f):
        if l in range(1,len(self.layers)):
            #Da inp costruisco la corrispondente lista di interi.
            valList = [inp.getValue(i) for i in range(inp.getLength())]

            #Calcolo gli outputs delle unità sui layers successivi.
            for i in range(1,l+1):
                valList = [unit.getOutput(valList,f) for unit in self.getLayer(i)]
            return valList
        else:
            raise RuntimeError ("getLayerOutput: no layer l found.")
    
    #Resituisce la lista degli output di rete (lista dei valori nelle unità di output) dato l'input inp.
    def getOutput(self, inp : Input):
        return self.getLayerOutput(inp,len(self.layers)-1, self.hyp['ActivFun'])
        
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
a3=Attribute(2,1)
i2=TRInput([a1,a2],True)
i3=Input([a1,a2,a3])

il=[i1,i2]
n = NeuralNetwork(il)
print(n.getOutput(i2))
print(n.getError([i2],0,1))
#ValueError: Inserted input is not valid for this NN!
#n.getOutput(i3)

