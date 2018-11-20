"""
Questa classe modella il concetto di attributo 
corrispondente ad uno specifico input. Chiamata 'N' la cardinalità del dominio dell'
attributo , il valore che un attributo può assumere va da 1 a N, quindi se avremo un 
attributo che può assumere i seguenti valori {"blue", "green", "red"}, il valore 1
corrisponderà a "blue", il valore 2 a "green" e il valore 3 a "red"
"""
class Attribute(object):

    #cardinality: cardinalità del dominio dell'attributo
    #value: valore da assegnare all'attributo
    def __init__(self, cardinality: int, value:int):
        if cardinality <= 0:
            raise ValueError ("cardinality must be greater than 0")

        if value < 1 or value > cardinality:
            raise ValueError (" value must be in this range: [1, cardinality]")

        #representation è la rappresentazione 1-of-k del valore di input
        self.represention = list()

        
        for i in range (0, cardinality):
            #memorizzo valori booleani perchè sono necessari solo 8 byte per rappresentarli
            if i == value-1:
                self.represention.append(True)
            else:
                self.represention.append(False)

    def copy(self):
        cardinality = len(self.represention)
        i = 0
        value : bool
        while i < len(self.represention):
            if self.represention[i]:
                value = i + 1
            i += 1
        return Attribute(cardinality, value)

    #restituisce il valore corrispondente a value all'interno del dominio
    #es. se representation = 010 ==> Decode(["blue", "green", "red"]) restituisce la stringa "green"
    def decode(self, dom: list):
        if len(dom) != len(self.represention):
            raise ValueError ("dom lenght must be equal to attribute cardinality")

        for value in self.represention:
            if value == 1:
                return dom[self.represention.index(value)]

    #interfacing: restituisce il numero di valori distinti che l'attributo corrente puo' assumere
    def getLength(self):
            return len(self.represention)

    #interfacing: restituisce il valore 0-1 dell'attributo in posizione interna i nella codifica 1-of-k
    def getValue(self, i: int):
            if i < 0 or i >= len(self.represention):
                raise ValueError ("Index i out of bounds")
            else:
                return self.represention[i]

    def print(self):
        string = ""
        for elem in self.represention:
            if elem:
                string += str(1)
            else:
                string += str(0)
        print(string)

    #restituisce la rappresentaziione come lista di interi
    def toInt(self):
        l = list()
        for elem in self.represention:
            l.append(int(elem))
        return l

class Input(object):

    def __init__(self, attributeList: list):
        if len(attributeList) <= 0:
            raise ValueError ("length of attribute list must be greater than 0")

        self.vector = list()

        for attribute in attributeList:
            if isinstance(attribute, Attribute):
                i = 0
                check = False
                while i < len(attribute.represention) and not check:
                    if attribute.represention[i]:
                        attr = Attribute(len(attribute.represention), i+1)
                        check = True
                        self.vector.append(attr)
                    i += 1
                
            else:
                raise ValueError ("attributeList must be a list of element of class Attribute")
    
    #interfacing: restituisce il numero di variabili 0-1 distinte dell'Input corrente
    def getLength(self):
        s = 0
        for attr in self.vector:
            s += attr.getLength()
        return s

    def len(self):
        return len(self.vector)

    #retituisce una lista di interi che è la codifica 1-of-k di tutti gli attributi dell'input
    def toInt(self):
        l = list()
        for attr in self.vector:
            l.append(attr.toInt())
        return l

    #interfacing: restituisce il valore 0-1 memorizzato in posizione i dell'Input corrente
    #ACHTUNG: i in (0,length-1)!!
    def getValue(self, i: int):
        if i >= 0 and i < self.getLength():
            j = i
            l = 0
            check = False
            while check == False:
                if j in range(0,self.vector[l].getLength()):
                    check = True
                    return self.vector[l].getValue(j)
                else:
                    j -= self.vector[l].getLength()
                    l += 1
        else:
            raise ValueError("Index i out of bounds")

    def print(self):
        for attr in self.vector:
            attr.print()

    def copy(self):
        attrList = list()
        for att in self.vector:
            attrList.append(att.copy())

        return Input(attrList)

    #get the attribute in position i inside vector
    def get(self, index:int):
        if(index < 0 or index > len(self.vector)):
            raise IndexError ("get: index out of bound")

        return self.vector[index]

    #restituisce un'unica lista contenente l'input codificato
    def getInput(self):
        l = list()
        for attribute in self.vector:
            l = l + attribute.copy().represention
        return l

#Sottoclasse degli input completi di target: ideali per TR, VS.
class TRInput(Input):
    def __init__(self, attributeList: list, target: bool):
        super().__init__(attributeList)
        self.target = target

    #interfacing: restituisce il target associato all'input corrente.
    def getTarget(self):
        return self.target

    def print(self):
        for attr in self.vector:
            attr.print()

        print("target "+ str(self.target))

    def copy(self):
        attrList = list()
        for att in self.vector:
            
            attrList.append(att.copy())

        return TRInput(attrList, self.target)

    def len(self):
        return len(self.vector)

    
            



#{"blue", "green", "red"}
a1 = Attribute(3, 2)
print(str(a1.decode(["blue", "green", "red"])))


#{-1, 0, 1, 2}
a2 = Attribute(4, 3)
print(str(a2.decode([-1, 0, 1, 2])))

attrList = [a1, a2]

input1 = Input(attrList)

input1.print()

print(input1.getValue(0)==0)
print(input1.getValue(1)==1)
print(input1.getValue(2)==0)
print(input1.getValue(3)==1)
print(input1.getValue(4)==0)
print(input1.getValue(5)==1)
print(input1.getValue(6)==1)

input2 = TRInput(attrList,True)
input2.print()


input3 = input2
print(input2 == input3)

input3 = input2.copy()
print(input2 == input3)

inp = input3.getInput()
l = list()
for elem in inp:
    l.append(int(elem))
print("input :"+ str(l))


# Output atteso:
# green
# 1
# 010
# 0010
# True
# True
# True
# False
# True
# True
# False
# 010
# 0010
# target True


