from Input import Attribute, Input, TRInput

"""
Classe, che dato un file in input, mi costruisce l'insieme dei dati di training
o di test utilizzati dalla rete neurale. I file avranno un pattern su ogni riga
e i vari attributi di ciascun pattern saranno separati da uno spazio 
"""

class DataSet(object):

    #filePath: percorso assoluto del file contenente i dati
    #tr: indica se il file contiene input con un target value (true) o no (false)
    #domains: lista di interi che indica la cardinalità dei domini dei vari attributi
    # (domains[i] = cardinalità del dominio dell'attributo i-esimo). 
    # Se nella lista è presente uno 0, significa che quell'attributo è da 
    # considerarsi come target value. Se nella lista è presente un numero < 0
    #significa che quell'attributo non va considerato
    #rowSkip: lista di interi indica eventuali righe da saltare (parte da 1)
    def __init__(self, filePath, splitchar, tr: bool , domains: list, rowSkip: list = None):
        
        
        try:
            f = open(filePath, "r")

            line = f.readline()
            i = 1
            target: bool
            isTargetFound = False
            attList = list()
            self.inputList = list()
            while line != "":
                if rowSkip == None or not i in rowSkip:
                    parsedLine = line.strip().split(splitchar)
                    for j in range(0, len(parsedLine)):
                        cardinality = domains[j]
                        
                        #target value
                        if cardinality == 0:
                            if tr:
                                if not isTargetFound:
                                    isTargetFound = True
                                    target = (int(parsedLine[j]) > 0)
                                else:
                                    raise RuntimeError("multiple target attribute in domains list")
                            else:
                                raise RuntimeError("tr is False but domains specifies that a target value exists")
                        
                        #attributo normale
                        if cardinality > 0:
                            attr = Attribute(cardinality, int(parsedLine[j]))
                            attList.append(attr)

                    if tr and not isTargetFound:
                        raise RuntimeError("tr is True, but no target value found")

                    isTargetFound = False

                    if tr:
                        inp = TRInput(attList, target)
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
        inputs = list()

        for inp in self.inputList:
            inputs.append(inp.copy())

        return inputs


domains = [0, 3, 3, 2, 3, 4, 2, -1]
dati = DataSet("C:\\Users\\matte\\Desktop\\Machine Learning\\monks-1.train", " ", True, domains)

for elem in dati.getInputs():
    elem.print()

skipRow = list()

for i in range(2,432):
    skipRow.append(i)

dati = DataSet("C:\\Users\\matte\\Desktop\\Machine Learning\\monks-1.train", " ", True, domains, skipRow)

print("prova skip")
for elem in dati.getInputs():
    elem.print()