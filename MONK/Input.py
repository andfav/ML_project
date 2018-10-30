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

    #restituisce il valore corrispondente a value all'interno del dominio
    #es. se representation = 010 ==> Decode(["blue", "green", "red"]) restituisce la stringa "green"
    def Decode(self, dom: list):
        if len(dom) != len(self.represention):
            raise ValueError ("dom lenght must be equal to attribute cardinality")

        for value in self.represention:
            if value == 1:
                return dom[self.represention.index(value)]

    def Print(self):
        string = ""
        for elem in self.represention:
            if elem:
                string += str(1)
            else:
                string += str(0)
        print(string)

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

    def Print(self):
        for attr in self.vector:
            attr.Print()

class TRInput(Input):

    def __init__(self, attributeList: list, target: bool):
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

        self.target = target

    def Print(self):
        for attr in self.vector:
            attr.Print()

        print("target "+ str(self.target))

#{"blue", "green", "red"}
a1 = Attribute(3, 2)
print(str(a1.Decode(["blue", "green", "red"])))

#{-1, 0, 1, 2}
a2 = Attribute(4, 3)
print(str(a2.Decode([-1, 0, 1, 2])))

attrList = [a1, a2]

input1 = Input(attrList)

input1.Print()


