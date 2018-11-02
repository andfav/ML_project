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

    def __init__(self, pos : int, dim : int, ValMax : float, prevLayer : list, f):
        from random import uniform
        #Inizializzazione dei pesi a valori casuali (distr. uniforme).
        self.weights = [uniform(0,ValMax) for i in range(dim)]

        #Memorizzazione della posizione nel layer corrente, del numero di connessioni 
        # al layer precedente, di copia del layer (lista) precedente, della funzione di attivazione.
        self.pos = pos
        self.dim = dim
        self.prevLayer = prevLayer
        self.activation = f
    
    #Restituisce il Net (somma pesata dei valori in ingresso all'unità) calcolato a partire
    #dall'input di rete inp.
    def getNet(self, inp: Input):
        s = 0
        if self.pos == 1:
            if inp.getLength() == self.dim:
                for i in range(inp.getLength()):
                    s += self.weights[i]*inp.getValue(i)
            else:
                raise ValueError ("Inserted input is not valid for this NN!")
        else:
            for unit in self.prevLayer:
                i = self.prevLayer.index(unit)
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

        # Inizializzo il layer successivo necessario al calcolo del delta.
        self.succLayer = []

    #Metodo che aggiunge il layer successivo. Per il fatto che il layer
    #successivo non esiste ancora in fase di inizializzazione dell'unità corrente
    #risulta necessario fare tale aggiunta in un momento successivo.
    def addSuccLayer(self, succLayer):
        self.succLayer = succLayer
    
    def getDelta(self):
        if len(self.succLayer) > 0:
            raise NotImplementedError
        else:
            raise RuntimeError ("A hidden unit has not constructed correctly: succLayer missing!")

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
                    'HiddenLayers': 1,
                    'HiddenUnits':  2,
                    'OutputUnits':  1, 
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

        #Inserimento in del training set.
        if len(trainingSet) == 0 or not isinstance(trainingSet[0], TRInput):
            raise ValueError ("inserted TR set is not valid!")
        else:
            length = trainingSet[0].getLength()
            for el in trainingSet:
                if not isinstance(el, TRInput) or el.getLength() != length:
                    raise ValueError ("TR set not valid: not homogeneous")
            self.layers.append(trainingSet.copy())

        #Creazione delle hidden units.
        #Costruzione del primo hidden layer: assumiamo che prevLayer = [] (il metodo getNet controlla
        # infatti pos e assume un comportamento diverso qualora pos==1).
        hiddenList = []
        for i in range(self.hyp['HiddenUnits']):
            hiddenList.append(HiddenUnit(1,length,self.hyp['ValMax'],[],self.hyp['ActivFun']))
        self.layers.append(hiddenList)
        length = len(hiddenList)
        #Costruzione degli hidden layers successivi (se esistono).
        for j in range(1,self.noHiddLayers):
            hiddenList = []
            for i in range(self.hyp['HiddenUnits']):
                hiddenList.append(HiddenUnit(j+1,length,self.hyp['ValMax'],self.getLayer(j),self.hyp['ActivFun']))
            self.layers.append(hiddenList)
            length = len(hiddenList)
            #Inserimento del succLayer nelle unità del layer precedente.
            for unit in self.getLayer(j):
                unit.addSuccLayer(self.getLayer(j+1))

        #Creazione delle output units.
        outputList = []
        for i in range(self.hyp['OutputUnits']):
                outputList.append(OutputUnit(self.noHiddLayers+1,length,self.hyp['ValMax'],self.getLayer(self.noHiddLayers),self.hyp['ActivFun']))
        self.layers.append(outputList)
        #Inserimento del succLayer nelle unità del layer precedente (hidden).
        for unit in self.getLayer(self.noHiddLayers):
            unit.addSuccLayer(self.getLayer(self.noHiddLayers+1))

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

    #Restituisce la lista degli output di rete dato un input.
    def getOutput(self, inp : Input):
        out_list = []
        for outnode in self.layers[len(self.layers)-1]:
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
a3=Attribute(2,1)
i2=TRInput([a1,a2],True)
i3=Input([a1,a2,a3])

il=[i1,i2]
n = NeuralNetwork(il)
print(n.getOutput(i2))
print(n.getError([i2],0,1))
#ValueError: Inserted input is not valid for this NN!
#n.getOutput(i3)

