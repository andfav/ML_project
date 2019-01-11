from Input import Input, TRInput, OneOfKAttribute, OneOfKInput, OneOfKTRInput
import numpy as np

"""
Classe, che dato un file in input, mi costruisce l'insieme dei dati di training
o di test utilizzati dalla rete neurale. I file avranno un pattern su ogni riga
e i vari attributi di ciascun pattern saranno separati da uno spazio 
"""

from enum import Enum
class ModeInput(Enum):
    INPUT = 1
    ONE_OF_K_INPUT = 2
    TR_INPUT = 3
    ONE_OF_K_TR_INPUT = 4

class DataSet(object):

    #filePath: percorso assoluto del file contenente i dati
    #mode: indica se il file contiene input con un target value (TR_INPUT o ONE_OF_K_TR_INPUT se è necessario codificarli)
    # o no (INPUT o ONE_OF_K_INPUT)
    #targetPos: posizione dell'attrinuto target, se presente (parte da 1)
    #domains: lista di interi che indica la cardinalità dei domini dei vari attributi
    # (domains[i] = cardinalità del dominio dell'attributo i-esimo) se è necessaria la codifica 1-of-k se l'attributo è un target la cardinalità sarà -1. 
    #rowSkip: lista di interi indica eventuali righe da saltare (parte da 1)
    #columnSkip: lista di interi indica eventuali attributi da saltare (parte da 1)
    def __init__(self, filePath, splitchar, mode: ModeInput, targetPos: list = None , domains: list = None, rowSkip: list = None, columnSkip = None):
        if (mode == ModeInput.ONE_OF_K_TR_INPUT or mode == ModeInput.TR_INPUT) and (targetPos == None):
            raise ValueError("Invalid target attribute position must be passed as parameter")
        
        if (mode == ModeInput.INPUT or mode == ModeInput.ONE_OF_K_INPUT) and (targetPos != None):
            raise ValueError("Invalid target attribute position must not be passed as parameter")

        try:
            f = open(filePath, "r")

            line = f.readline()
            i = 1
            
            attList = list()
            self.inputList = list()
            while line != "":
                if rowSkip == None or not i in rowSkip:
                    parsedLine = line.strip().split(splitchar)

                    for pos in targetPos:
                        if pos > len(parsedLine):
                            raise ValueError("Invalid target attribute position (Out of Bound)")

                    #creo array target
                    target = list()
                    for pos in targetPos:
                        if mode == ModeInput.ONE_OF_K_TR_INPUT:
                            elem = int(parsedLine[pos-1])
                        elif mode == ModeInput.TR_INPUT:
                            elem = float(parsedLine[pos-1])
                        target.append(elem)
                    targetArray = np.array(target)


                    for j in range(0, len(parsedLine)):
                        if columnSkip == None or not (j+1) in columnSkip:
                            if j+1 not in targetPos:
                                if mode == ModeInput.ONE_OF_K_INPUT or mode == ModeInput.ONE_OF_K_TR_INPUT:
                                    cardinality = domains[j]
                                    #attributo normale
                                    if cardinality > 0:
                                        attr = OneOfKAttribute(cardinality, int(parsedLine[j]))
                                        attList.append(attr)
                                    else:
                                        raise ValueError("all not target domains value must be > 0")
                                
                                elif mode == ModeInput.INPUT or mode == ModeInput.TR_INPUT:
                                    attr = float(parsedLine[j])
                                    attList.append(attr)

                    if mode == ModeInput.TR_INPUT:
                        inp = TRInput(attList, targetArray)
                        self.inputList.append(inp)

                    elif mode == ModeInput.ONE_OF_K_TR_INPUT:
                        inp = OneOfKTRInput(attList, targetArray)
                        self.inputList.append(inp)

                    elif mode == ModeInput.ONE_OF_K_INPUT:
                        inp = OneOfKInput(attList)
                        self.inputList.append(inp)

                    else:
                        inp = Input(attList)
                        self.inputList.append(inp)
                    attList.clear()
                
                i += 1
                line = f.readline()

            f.close()
        except IOError:
            raise

    #restituisce la lista degli input
    def getInputs(self):
        return self.inputList

    """
    Funzione che restringe i valori di tutti gli attributi e i target nell'intervallo [v1, v2]
    """
    def restrict(self, v1, v2):
        if v1 >= v2:
            raise ValueError("v1 must be smaller than v2")

        if isinstance (self.inputList[0], OneOfKInput):
            raise RuntimeError("cannot apply to one-of-k encoded input") 

       
        minAttr = list() #contiene i valori minimi di tutti gli attributi
        maxAttr = list() #contiene i valori massimi di tutti gli attributi
        

        #inizializzo minAttr, maxAttr
        for attr in self.inputList[0].getInput():
            minAttr.append(attr)
            maxAttr.append(attr)

        
        ###################################################

        for i in range(1, len(self.inputList)):
            attrArray = self.inputList[i].getInput()
            for j in range(len(attrArray)):
                minAttr[j] = min(minAttr[j], attrArray[j])
                maxAttr[j] = max(maxAttr[j], attrArray[j])


        minAttrArray = np.array(minAttr)
        maxAttrArray = np.array(maxAttr)
        

        for inp in self.inputList:
            attrArray = inp.getInput()
            attrArray = ((attrArray - minAttrArray) / (maxAttrArray - minAttrArray)) * (v2-v1) + v1*(maxAttrArray - minAttrArray) / (maxAttrArray - minAttrArray)
            inp.representation = attrArray
            
        else:
            return (minAttrArray, maxAttrArray)

"""
domains = [-1, 3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = [1]
dati = DataSet("C:\\Users\\matte\\Desktop\\Machine Learning\\monks-1.train", " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)

print("*************************\n\nDataset")
for elem in dati.getInputs():
    elem.print()

skipRow = list()

for i in range(2,432):
    skipRow.append(i)

dati = DataSet("C:\\Users\\matte\\Desktop\\Machine Learning\\monks-1.train", " ", ModeInput.ONE_OF_K_TR_INPUT,  targetPos, domains, skipRow, columnSkip)

print("prova skip")
for elem in dati.getInputs():
    elem.print()

"""

skipRow = [1,2,3,4,5,6,7,8,9,10]
columnSkip = [1]
target = [12,13]

dati = DataSet("ML-CUP18-TR.csv", ",", ModeInput.TR_INPUT, target, None, skipRow, columnSkip)

print("*************************\n\nDataset")
for elem in dati.getInputs():
    elem.print()

dati.restrict(-1, 1)

print("*************************\n\nDataset normlizzato")
for elem in dati.getInputs():
    elem.print()