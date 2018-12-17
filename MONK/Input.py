import numpy as np

"""
Questa classe modella il concetto di attributo 
corrispondente ad uno specifico input. Chiamata 'N' la cardinalità del dominio dell'
attributo , il valore che un attributo può assumere va da 1 a N, quindi se avremo un 
attributo che può assumere i seguenti valori {"blue", "green", "red"}, il valore 1
corrisponderà a "blue", il valore 2 a "green" e il valore 3 a "red"
"""
class OneOfKAttribute(object):

    #cardinality: cardinalità del dominio dell'attributo
    #value: valore da assegnare all'attributo
    def __init__(self, cardinality: int, value:int):
        if cardinality <= 0:
            raise ValueError ("cardinality must be greater than 0")

        if value < 1 or value > cardinality:
            raise ValueError (" value must be in this range: [1, cardinality]")

        #representation è la rappresentazione 1-of-k del valore di input
        l = list()

        
        for i in range (0, cardinality):
            #memorizzo valori booleani perchè sono necessari solo 8 byte per rappresentarli
            if i == value-1:
                l.append(1)
            else:
                l.append(0)

        self.representation = l

    #interfacing: restituisce il numero di valori distinti che l'attributo corrente puo' assumere
    def getLength(self):
            return len(self.representation)

    #interfacing: restituisce il valore 0-1 dell'attributo in posizione interna i nella codifica 1-of-k
    def getValue(self, i: int):
            if i < 0 or i >= len(self.representation):
                raise ValueError ("Index i out of bounds")
            else:
                return self.representation[i]

    def print(self):
        string = ""
        for elem in self.representation:
            if elem:
                string += str(1)
            else:
                string += str(0)
        print(string)


    def getRepresentation(self):
        return self.representation
"""
Input i cui attributi possono asumere valori continui
"""
class Input(object):

    def __init__(self, attributeList: list):
        if len(attributeList) <= 0:
            raise ValueError ("length of attribute list must be greater than 0")

        self.representation = np.array(attributeList)

    #interfacing: restituisce il numero di attributi dell'Input corrente
    def getLength(self):
        return len(self.representation)

    #interfacing: restituisce il valore dell'attributo i-esimo
    def getValue(self, i: int):
        if i >= 0 and i < len(self.representation):
            return self.representation[i]
        else:
            raise ValueError("Index i out of bounds")

    def print(self):
        string = ""
        i = 0
        while i < (len(self.representation) -1):
            string+= str(self.representation[i]) + " "
            i += 1
        string+= str(self.representation[i])
        print(string)

    def getInput(self):
        return self.representation
"""
Input i cui attributi sono codificati in modo 1-of-k
"""
class OneOfKInput(Input):

    def __init__(self, attributeList: list):
        if len(attributeList) <= 0:
            raise ValueError ("length of attribute list must be greater than 0")

        l = list()
        
        for i in range (len(attributeList)):
            if isinstance(attributeList[i], OneOfKAttribute):
                l = l + attributeList[i].getRepresentation()
            else:
                raise ValueError ("attributeList must be a list of element of class OneOfKAttribute")

        self.representation = np.array(l)
    #interfacing: restituisce il numero di variabili 0-1 distinte dell'Input corrente
    def getLength(self):
        return len(self.representation)

    #interfacing: restituisce il valore 0-1 memorizzato in posizione i dell'Input corrente
    #ACHTUNG: i in (0,length-1)!!
    def getValue(self, i: int):
        if i >= 0 and i < self.getLength():
            return self.representation[i]
        else:
            raise ValueError("Index i out of bounds")

    def print(self):
        super().print()

    #restituisce un'unica lista contenente l'input codificato
    def getInput(self):
        return self.representation

"""
Sottoclasse degli input completi di target: ideali per TR, VS.
"""
class TRInput(Input):
    def __init__(self, attributeList: list, target):
        super().__init__(attributeList)
        self.target = target

    #interfacing: restituisce il target associato all'input corrente.
    def getTarget(self):
        return self.target

    def getTargetSigmoidal(self):
        return min(max(self.target,0.1),0.9)

    def print(self):
        super().print()
        print("target "+ str(self.target))

    def getLength(self):
        return super().getLength()
            

"""
Sottoclasse degli input completi di target: ideali per TR, VS.
"""
class OneOfKTRInput(OneOfKInput):
    def __init__(self, attributeList: list, target):
        super().__init__(attributeList)
        self.target = target

    #interfacing: restituisce il target associato all'input corrente.
    def getTarget(self):
        return self.target
    
    def getTargetSigmoidal(self):
        return min(max(self.target,0.1),0.9)

    def print(self):
        super().print()
        print("target "+ str(self.target))

    
    def getLength(self):
        return super().getLength()

"""
#{"blue", "green", "red"}
a1 = OneOfKAttribute(3, 2)



#{-1, 0, 1, 2}
a2 = OneOfKAttribute(4, 3)




input1 = OneOfKInput([a1, a2])

input1.print()

print(input1.getValue(0)==0)
print(input1.getValue(1)==1)
print(input1.getValue(2)==0)
print(input1.getValue(3)==1)
print(input1.getValue(4)==0)
print(input1.getValue(5)==1)
print(input1.getValue(6)==1)

input2 = OneOfKTRInput([a1, a2], 1)
input2.print()

trInput1 = TRInput([1, 2, 4, 6.1, 6753.0545], 23)

trInput1.print()
"""
"""
Output atteso:
0 1 0 0 0 1 0
# True
# True
# True
# False
# True
# True
# False
target 1
1.0 2.0 4.0 6.1 6753.0545
target 23
"""

