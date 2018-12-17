from Input import Input, TRInput, OneOfKAttribute, OneOfKInput, OneOfKTRInput

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
    # (domains[i] = cardinalità del dominio dell'attributo i-esimo) se è necessaria la codifica 1-of-k. 
    #rowSkip: lista di interi indica eventuali righe da saltare (parte da 1)
    #columnSkip: lista di interi indica eventuali attributi da saltare (parte da 1)
    def __init__(self, filePath, splitchar, mode: ModeInput, targetPos = 0 , domains: list = None, rowSkip: list = None, columnSkip = None):
        if (mode == ModeInput.ONE_OF_K_TR_INPUT or mode == ModeInput.TR_INPUT) and (targetPos <= 0):
            raise ValueError("Invalid target attribute position (must be > 0)")
        
        if (mode == ModeInput.INPUT or mode == ModeInput.ONE_OF_K_INPUT) and (targetPos > 0):
            raise ValueError("Invalid target attribute position (must be <= 0)")

        try:
            f = open(filePath, "r")

            line = f.readline()
            i = 1
            
            attList = list()
            self.inputList = list()
            while line != "":
                if rowSkip == None or not i in rowSkip:
                    parsedLine = line.strip().split(splitchar)

                    if targetPos > len(parsedLine):
                        raise ValueError("Invalid target attribute position (Out of Bound)")

                    for j in range(0, len(parsedLine)):
                        if columnSkip == None or not (j+1) in columnSkip:
                            attrPos: int
                            if (targetPos-1) > j:
                                attrPos = j
                            else:
                                attrPos = j-1

                            #target value
                            if targetPos == (j+1):
                                if mode == ModeInput.ONE_OF_K_TR_INPUT:
                                    target = int(parsedLine[j])
                                elif mode == ModeInput.TR_INPUT:
                                    target = float(parsedLine[j])

                            else:
                                
                                if mode == ModeInput.ONE_OF_K_INPUT or mode == ModeInput.ONE_OF_K_TR_INPUT:
                                    cardinality = domains[attrPos]
                                    #attributo normale
                                    if cardinality > 0:
                                        attr = OneOfKAttribute(cardinality, int(parsedLine[j]))
                                        attList.append(attr)
                                    else:
                                        raise ValueError("all domains value must be > 0")
                                
                                elif mode == ModeInput.INPUT or mode == ModeInput.TR_INPUT:
                                    attr = float(parsedLine[j])
                                    attList.append(attr)

                    if mode == ModeInput.TR_INPUT:
                        inp = TRInput(attList, target)
                        self.inputList.append(inp)

                    elif mode == ModeInput.ONE_OF_K_TR_INPUT:
                        inp = OneOfKTRInput(attList, target)
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
domains = [3, 3, 2, 3, 4, 2]
columnSkip = [8]
targetPos = 1
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
