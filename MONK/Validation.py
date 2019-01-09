from DataSet import DataSet, ModeInput
from NN import NeuralNetwork, ModeLearn
from ActivFunct import ActivFunct, Sigmoidal, Identity
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random as rnd
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as graphic

def k_fold_CV_single(k: int, dataSet, f:ActivFunct, theta, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if k <= 0:
        raise ValueError ("Wrong value of num. folders inserted")

    cp = dataSet.copy()
    
    #Rimescolo il data set.
    rnd.shuffle(cp)

    #Costruisco la sottolista dei dati divisibile esattamente per k.
    h = len(cp) - len(cp) % k
    dataSetExact = cp[0:h]

    #Creo la lista dei folders.
    folderDim = int(len(dataSetExact) / k)
    folders = [cp[i*folderDim : (i+1)*folderDim] for i in range(k)]

    #Inserisco gli elementi di avanzo.
    for i in range(len(cp)-h):
        folders[i].append(cp[i+h])

    errore = list()

    #per stampare l'errore sul traininig set e sul validation set
    trErrorPlot = list()
    vlErrorPlot = list()

    for i in range (len(folders)):
        lcopy = folders.copy()
        del(lcopy[i])

        vlSet = folders[i]
        trSet = list()
        for j in range (len (lcopy)):
            trSet+= lcopy[j]
        nn = NeuralNetwork(trSet, f, theta)
        (trErr, vlErr, trAcc, vlAcc) = nn.learn(modeLearn, errorFunct, miniBatchDim, vlSet)
        trErrorPlot.append(trErr)
        vlErrorPlot.append(vlErr)
        errore.append(nn.getError(vlSet, 0, 1/len(vlSet), errorFunct))

    err = sum(errore)/k

    #controllo che tutti gli errorPlot abbiano la stessa lunghezza
    maxLen = len(trErrorPlot[0])
    for i in range(1, len(trErrorPlot)):
        if len(trErrorPlot[i]) > maxLen:
            maxLen = len(trErrorPlot[i])

    for i in range(len(trErrorPlot)):
        if len(trErrorPlot[i]) < maxLen:
            for j in range(maxLen-len(trErrorPlot[i])):
                trErrorPlot[i].append(trErrorPlot[i][-1])
                vlErrorPlot[i].append(vlErrorPlot[i][-1])



    trErrorArray = np.array(trErrorPlot[0])
    vlErrorArray = np.array(vlErrorPlot[0])

    
    for i in range (1, len(trErrorPlot)):
        trErrorArray = trErrorArray + np.array(trErrorPlot[i])
    trErrorArray = trErrorArray / k

    for i in range (1, len(vlErrorPlot)):
        vlErrorArray = vlErrorArray + np.array(vlErrorPlot[i])
    vlErrorArray = vlErrorArray / k
    
    return (err, trErrorArray, vlErrorArray)
    
    

def cross_validation(workers: int, nFolder:int, modeLearn:ModeLearn, dataSet, f:ActivFunct, errorFunct:ActivFunct, learnRate:list, momRate:list, regRate:list, ValMax:list, HiddenUnits:list, OutputUnits:list, MaxEpochs:list, Tolerance:list, startTime, miniBatchDim= None):
    if workers <= 0:
        workers = 1
    
    dictionaries = list()
 
    #generazione di tutte le possibili combinazioni
    for eta in learnRate:
        for alfa in momRate:
            for lambd in regRate:
                for maxVal in ValMax:
                    for hUnit in HiddenUnits:
                        for oUnit in OutputUnits:
                            for epochs in MaxEpochs:
                                for epsilon in Tolerance:
                                    theta = {'learnRate':eta, 'momRate':alfa, 'regRate':lambd, 'ValMax':maxVal, 'HiddenUnits':hUnit, 'OutputUnits':oUnit, 'MaxEpochs':epochs, 'Tolerance':epsilon}
                                    dictionaries.append(theta)

    totalTheta = len(dictionaries)
    print("Configurazioni totali di iperparametri: "+str(totalTheta))

    res = list()
    future = list()
    with ProcessPoolExecutor(max_workers=min(workers,totalTheta)) as pool:
        for theta in dictionaries:
            future.append(pool.submit(partial(cross_validation_iterator, nIter=10, workers=workers, nFolder=nFolder, dataSet=dataSet.copy(), f=f, theta=theta, startTime=startTime, errorFunct=errorFunct, modeLearn=modeLearn, miniBatchDim=miniBatchDim)))

        for elem in future:
            res.append(elem.result())

    return res
    
def cross_validation_iterator(workers: int, nFolder: int, dataSet, f:ActivFunct, theta:dict, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None, nIter: int = 10):
    errore = list()
    trErrorPlot = list()
    vlErrorPlot = list()
    for i in range(0, nIter):
        
        (err, trErr, vlErr) = k_fold_CV_single(nFolder,dataSet,f,theta,errorFunct,modeLearn,miniBatchDim)
        errore.append(err)
        trErrorPlot.append(trErr.tolist())
        vlErrorPlot.append(vlErr.tolist())

    endTime = time.time()
    print("************")
    print("Configurazione completata, tempo: " + str(endTime - startTime))
    print(str((theta, sum(errore)/nIter)))

    #controllo che tutti gli errorPlot abbiano la stessa lunghezza
    maxLen = len(trErrorPlot[0])
    for i in range(1, len(trErrorPlot)):
        if len(trErrorPlot[i]) > maxLen:
            maxLen = len(trErrorPlot[i])

    for i in range(len(trErrorPlot)):
        if len(trErrorPlot[i]) < maxLen:
            for j in range(maxLen-len(trErrorPlot[i])):
                trErrorPlot[i].append(trErrorPlot[i][-1])
                vlErrorPlot[i].append(vlErrorPlot[i][-1])

    
    trArray = np.array(trErrorPlot[0])
    vlArray = np.array(vlErrorPlot[0])

    
    for i in range (1,nIter):
        trArray = trArray + np.array(trErrorPlot[i])
        vlArray = vlArray + np.array(vlErrorPlot[i])

    trArray = trArray / nIter
    vlArray = vlArray / nIter

    return (theta, sum(errore)/nIter, trArray, vlArray)
    

"""
Funzione che implementa la double cross validation.
Argomenti:
- workers, numero massimo di processi attivi;
- testFolder, numero di folder in cui dividere inizialmente il dataSet (testSet e validation + training);
- nFolder, numero di folder di ciascuna k-fold validation;
- dataSet, il data set;
- f, funzione di attivazione della rete neurale;
- theta, diziorario di tutti gli iperparametri;
- errorFunct, funzione di valutazione dell'errore;
"""

def double_cross_validation(workers: int, testFolder: int, nFolder: int, dataSet, f:ActivFunct, learnRate:list, momRate:list, regRate:list, ValMax:list, HiddenUnits:list, OutputUnits:list, MaxEpochs:list, Tolerance:list, startTime, errorFunct = None, modeLearn:ModeLearn = ModeLearn.BATCH, miniBatchDim= None):
    if testFolder <= 1:
        raise ValueError ("Wrong value of num. folders inserted")

    cp = dataSet.copy()
    
    #Rimescolo il data set.
    rnd.shuffle(cp)

    #Costruisco la sottolista dei dati divisibile esattamente per testFolder.
    h = len(cp) - len(cp) % testFolder
    dataSetExact = cp[0:h]

    #Creo la lista dei folders.
    folderDim = int(len(dataSetExact) / testFolder)
    folders = [cp[i*folderDim : (i+1)*folderDim] for i in range(testFolder)]

    #Inserisco gli elementi di avanzo.
    for i in range(len(dataSet)-h):
        folders[i].append(cp[i+h])

    errList = list()
    for i in range(len(folders)):
        foldersCopy = folders.copy()
        testSet = foldersCopy[i]
        del(foldersCopy[i])

        vlSet = list()
        for j in range(len(foldersCopy)):
            vlSet += foldersCopy[j]

        e = cross_validation(workers,nFolder,modeLearn,vlSet,f,errorFunct,learnRate,momRate,regRate,ValMax,HiddenUnits,OutputUnits,MaxEpochs,Tolerance,startTime,miniBatchDim=miniBatchDim)
        theta = getBestResult(e)[0]
        nn = NeuralNetwork(vlSet,f,new_hyp=theta)
        (_,testErr,_,_)=nn.learn(modeLearn,errorFunct,miniBatchDim,testSet)
        errList.append(testErr[-1])

    return 1/testFolder * sum(errList)



def getBestResult(e):
    l = [e[i][1] for i in range(len(e))]
    return e[l.index(min(l))]

if __name__ == '__main__':
    """
    Test della cross-validation
    """
    
    f = Sigmoidal(12)

    domains = [3, 3, 2, 3, 4, 2]
    columnSkip = [8]
    targetPos = 1
    datasetFileName = "Z:\Matteo\Desktop\Machine Learning\monks-3.train"
    logFileName = datasetFileName + "sigm12.log"

    trainingSet = DataSet(datasetFileName, " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
    learnRates = [0.1]
    momRates = [0.5, 0.6, 0.7]
    regRates = [0.007]
    ValMaxs = [0.7]
    HiddenUnitss = [2,3,4]
    OutputUnitss = [1]
    MaxIterss = [600]
    Tolerances = [0.03]
    start = time.time()
    e = cross_validation(9, 10, ModeLearn.BATCH, trainingSet.getInputs(), f, None, learnRates, momRates, regRates, ValMaxs, HiddenUnitss, OutputUnitss, MaxIterss, Tolerances, start, None)
    stop = time.time() 
    secdiff = stop - start

    log = open(logFileName,'a')
    log.write("***\n")
    log.write("File: " + datasetFileName + ", con configurazioni di iperparametri seguenti \n")
    log.write("learnRates: " + str(learnRates))
    log.write('\n')
    log.write("momRates: " + str(momRates))
    log.write('\n')
    log.write("regRates: " + str(regRates))
    log.write('\n')
    log.write("ValMaxs: " + str(ValMaxs))
    log.write('\n')
    log.write("HiddenUnitss: " + str(HiddenUnitss))
    log.write('\n')
    log.write("OutputUnitss: " + str(OutputUnitss))
    log.write('\n')
    log.write("MaxIterss: " + str(MaxIterss))
    log.write('\n')
    log.write("Tolerances: " + str(Tolerances))
    log.write('\n\n')

    log.write("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi, risultati:\n")

    log.write(str(e) + '\n')
    log.write('\n')
    log.write("Miglior risultato: " + str(getBestResult(e)) + '\n')
    log.write('\n\n')

    log.close()

    
    for elem in e:
        graphic.plot(elem[2], 'r--')#tr set
        graphic.plot(elem[3], 'b')#vl set
        graphic.show()
    
    
    print("Operazione di cross-validation conclusa in " + str(secdiff) + " secondi: i dati sono stati salvati in " + logFileName + ".")
    
    """
    trError = np.array([0.16610955, 0.16094129, 0.15395562, 0.14815347, 0.14280768,
       0.1376045 , 0.13209849, 0.12650345, 0.12088108, 0.11538641,
       0.11023716, 0.10553905, 0.10138383, 0.09778859, 0.09472032,
       0.09212012, 0.08991899, 0.08804299, 0.08643415, 0.0850429 ,
       0.08382449, 0.08274238, 0.08176475, 0.08086468, 0.08002181,
       0.0792202 , 0.07844698, 0.07769254, 0.07694944, 0.07621163,
       0.07547429, 0.07473349, 0.07398604, 0.07322944, 0.07246174,
       0.07168143, 0.07088739, 0.07007891, 0.06925556, 0.0684172 ,
       0.06756396, 0.06669617, 0.06581433, 0.06491904, 0.06401088,
       0.0630903 , 0.06215762, 0.06121301, 0.06025675, 0.05928976,
       0.05831436, 0.05733401, 0.05635174, 0.05536919, 0.05438738,
       0.05340737, 0.05243071, 0.05145936, 0.05049511, 0.04953903,
       0.04859124, 0.04765111, 0.04671775, 0.04579053, 0.04486958,
       0.043956  , 0.04305142, 0.04215748, 0.04127571, 0.04040753,
       0.0395544 , 0.03871787, 0.03789958, 0.0371012 , 0.03632373,
       0.03556714, 0.03483059, 0.0341129 , 0.03341273, 0.03272875,
       0.03205974, 0.03140458, 0.03076212, 0.03013122, 0.02951077,
       0.02890011, 0.02830014, 0.02771309, 0.02713919, 0.02657548,
       0.02601897, 0.02546946, 0.02493235, 0.02441271, 0.02390735,
       0.02341377, 0.02293129, 0.02245944, 0.02199794, 0.02154647,
       0.02110483, 0.02067287, 0.02025056, 0.01983808, 0.01943584,
       0.01904461, 0.0186655 , 0.01829979, 0.0179485 , 0.01761221,
       0.01729095, 0.0169844 , 0.01669212, 0.01641371, 0.0161488 ,
       0.01589705, 0.01565811, 0.01543161, 0.01521704, 0.01501363,
       0.01482037, 0.01463617, 0.01446006, 0.01429141, 0.01412992,
       0.01397553, 0.01382801, 0.01368688, 0.01355153, 0.01342126,
       0.01329542, 0.01317332, 0.01305426, 0.01293744, 0.01282194,
       0.01270675, 0.01259108, 0.01247496, 0.01235952, 0.01224603,
       0.01213656, 0.01203403, 0.01193891, 0.01184982, 0.01176579,
       0.01168604, 0.0116097 , 0.01153607, 0.0114647 , 0.01139535,
       0.01132784, 0.01126205, 0.01119793, 0.01113548, 0.01107477,
       0.0110159 , 0.01095904, 0.01090434, 0.01085195, 0.01080199,
       0.01075449, 0.01070945, 0.01066676, 0.01062631, 0.01058794,
       0.01055147, 0.01051674, 0.01048357, 0.0104518 , 0.01042126,
       0.01039181, 0.01036329, 0.01033555, 0.01030844, 0.01028181,
       0.01025551, 0.01022941, 0.01020338, 0.01017727, 0.01015095,
       0.0101243 , 0.01009712, 0.01006922, 0.01004038, 0.0100105 ,
       0.00997995, 0.00994994, 0.00992224, 0.00989787, 0.00987665,
       0.00985793, 0.00984118, 0.00982606, 0.00981235, 0.00979984,
       0.00978837, 0.00977783, 0.00976811, 0.00975915, 0.00975085,
       0.00974316, 0.00973599, 0.00972925, 0.00972287, 0.00971676,
       0.00971085, 0.00970504, 0.00969927, 0.00969345, 0.00968749,
       0.00968132, 0.00967482, 0.00966789, 0.0096604 , 0.00965215,
       0.00964288, 0.00963221, 0.00961952, 0.00960403, 0.00958546,
       0.00956616, 0.0095493 , 0.00953399, 0.00951893, 0.00950337,
       0.00948679, 0.00946897, 0.00944978, 0.00942891, 0.00940585,
       0.00937991, 0.00935097, 0.00932088, 0.00929207, 0.0092648 ,
       0.00923905, 0.00921537, 0.00919409, 0.00917466, 0.00915638,
       0.00913873, 0.00912142, 0.00910425, 0.00908706, 0.0090697 ,
       0.00905205, 0.009034  , 0.00901545, 0.00899635, 0.00897667,
       0.00895638, 0.00893552, 0.00891413, 0.00889229, 0.00887009,
       0.00884764, 0.00882505, 0.00880245, 0.00877991, 0.00875749,
       0.00873517, 0.00871289, 0.0086905 , 0.0086678 , 0.00864464,
       0.00862102, 0.00859717, 0.00857335, 0.0085496 , 0.0085256 ,
       0.00850086, 0.00847486, 0.00844723, 0.00841783, 0.00838724,
       0.00835697, 0.00832867, 0.00830322, 0.00828058, 0.00826051,
       0.00824285, 0.0082276 , 0.00821471, 0.00820402, 0.00819532,
       0.00818837, 0.008183  , 0.00817903, 0.0081763 , 0.00817466,
       0.00817393, 0.008174  , 0.00817472, 0.00817601, 0.00817778,
       0.00817996, 0.00818251, 0.00818537, 0.00818852, 0.00819191,
       0.00819551, 0.00819929, 0.0082032 , 0.00820721, 0.00821127,
       0.00821534, 0.00821939, 0.00822336, 0.00822721, 0.00823088,
       0.00823431, 0.00823746, 0.00824027, 0.00824269, 0.00824468,
       0.00824622, 0.00824734, 0.00824806, 0.00824846, 0.00824863,
       0.00824869, 0.00824874, 0.00824885, 0.00824911, 0.00824957,
       0.00825026, 0.00825119, 0.0082524 , 0.00825387, 0.0082556 ,
       0.00825759, 0.00825982, 0.00826229, 0.00826497, 0.00826784,
       0.0082709 , 0.00827413, 0.00827753, 0.00828108, 0.00828478,
       0.00828862, 0.00829261, 0.00829674, 0.008301  , 0.00830539,
       0.00830992, 0.00831457, 0.00831933, 0.00832421, 0.00832918,
       0.00833426, 0.00833941, 0.00834465, 0.00834996, 0.00835533,
       0.00836076, 0.00836624, 0.00837176, 0.00837732, 0.00838292,
       0.00838854, 0.0083942 , 0.00839987, 0.00840557, 0.00841128,
       0.00841701, 0.00842276, 0.00842852, 0.00843429, 0.00844006,
       0.00844585, 0.00845164, 0.00845744, 0.00846325, 0.00846906,
       0.00847487, 0.00848068, 0.00848649, 0.0084923 , 0.00849812,
       0.00850392, 0.00850973, 0.00851552, 0.00852132, 0.0085271 ,
       0.00853287, 0.00853864, 0.00854439, 0.00855013, 0.00855585,
       0.00856157, 0.00856726, 0.00857294, 0.0085786 , 0.00858425,
       0.00858987, 0.00859548, 0.00860106, 0.00860663, 0.00861217,
       0.00861769, 0.00862318, 0.00862866, 0.0086341 , 0.00863953,
       0.00864492, 0.00865029, 0.00865563, 0.00866095, 0.00866623,
       0.00867149, 0.00867671, 0.00868191, 0.00868707, 0.0086922 ,
       0.00869729, 0.00870234, 0.00870736, 0.00871234, 0.00871728,
       0.00872217, 0.00872702, 0.00873183, 0.00873658, 0.00874129,
       0.00874594, 0.00875053, 0.00875506, 0.00875952, 0.00876392,
       0.00876824, 0.00877249, 0.00877665, 0.00878072, 0.0087847 ,
       0.00878858, 0.00879234, 0.00879599, 0.00879952, 0.00880292,
       0.00880617, 0.00880928, 0.00881224, 0.00881504, 0.00881768,
       0.00882015, 0.00882246, 0.0088246 , 0.00882658, 0.00882841,
       0.0088301 , 0.00883168, 0.00883315, 0.00883454, 0.00883588,
       0.00883719, 0.00883849, 0.00883981, 0.00884116, 0.00884256,
       0.00884402, 0.00884557, 0.0088472 , 0.00884892, 0.00885074,
       0.00885267, 0.0088547 , 0.00885683, 0.00885907, 0.00886141,
       0.00886385, 0.00886638, 0.00886901, 0.00887173, 0.00887453,
       0.00887741, 0.00888036, 0.00888339, 0.00888648, 0.00888964,
       0.00889285, 0.00889611, 0.00889943, 0.00890279, 0.00890619,
       0.00890963, 0.00891311, 0.00891662, 0.00892016, 0.00892372,
       0.00892731, 0.00893092, 0.00893455, 0.0089382 , 0.00894186,
       0.00894554, 0.00894923, 0.00895293, 0.00895664, 0.00896036,
       0.00896408, 0.00896782, 0.00897155, 0.00897529, 0.00897903,
       0.00898277, 0.00898652, 0.00899026, 0.00899401, 0.00899775,
       0.00900149, 0.00900523, 0.00900897, 0.0090127 , 0.00901643,
       0.00902015, 0.00902387, 0.00902759, 0.0090313 , 0.009035  ,
       0.0090387 , 0.00904239, 0.00904607, 0.00904975, 0.00905342,
       0.00905709, 0.00906075, 0.0090644 , 0.00906804, 0.00907167,
       0.0090753 , 0.00907892, 0.00908253, 0.00908613, 0.00908972,
       0.00909331, 0.00909688, 0.00910045, 0.00910401, 0.00910756,
       0.0091111 , 0.00911463, 0.00911815, 0.00912167, 0.00912517,
       0.00912867, 0.00913215, 0.00913563, 0.0091391 , 0.00914256,
       0.00914601, 0.00914945, 0.00915288, 0.0091563 , 0.00915971,
       0.00916312, 0.00916651, 0.0091699 , 0.00917327, 0.00917664,
       0.00918   , 0.00918334, 0.00918668, 0.00919001, 0.00919333,
       0.00919664, 0.00919994, 0.00920324, 0.00920652, 0.00920979,
       0.00921306, 0.00921632, 0.00921956, 0.0092228 , 0.00922603,
       0.00922925, 0.00923246, 0.00923566, 0.00923886, 0.00924204,
       0.00924521, 0.00924838, 0.00925154, 0.00925469, 0.00925783,
       0.00926096, 0.00926408, 0.0092672 , 0.0092703 , 0.0092734 ,
       0.00927649, 0.00927956, 0.00928264, 0.0092857 , 0.00928875,
       0.0092918 , 0.00929483, 0.00929786, 0.00930088, 0.0093039 ])

    vlError = np.array([0.19281776, 0.16939123, 0.16650974, 0.15918293, 0.15351606,
       0.15048286, 0.14615938, 0.14191631, 0.13823212, 0.13403832,
       0.13024753, 0.12690895, 0.12377456, 0.12119668, 0.11909823,
       0.11726108, 0.11572254, 0.11440439, 0.1132413 , 0.11221895,
       0.11135417, 0.11061841, 0.10998834, 0.10945389, 0.10899482,
       0.1085921 , 0.1082359 , 0.10792062, 0.10763972, 0.10738887,
       0.10716583, 0.10696655, 0.10678649, 0.10662163, 0.10646735,
       0.10631876, 0.10617106, 0.10601915, 0.10585755, 0.10568056,
       0.10548252, 0.10525826, 0.10500356, 0.10471555, 0.10439278,
       0.1040352 , 0.10364383, 0.1032206 , 0.10276821, 0.10229008,
       0.1017898 , 0.10126955, 0.10072798, 0.10016117, 0.09956776,
       0.09895123, 0.09831636, 0.09766606, 0.09700066, 0.09631823,
       0.09561591, 0.09489082, 0.09414059, 0.09336352, 0.0925584 ,
       0.09172419, 0.09085997, 0.08996535, 0.08904084, 0.08808743,
       0.08710609, 0.08609845, 0.08506793, 0.08402058, 0.08296458,
       0.08190789, 0.08085499, 0.07980595, 0.07875869, 0.07771214,
       0.07666746, 0.07562676, 0.07459101, 0.07355881, 0.07252652,
       0.07148911, 0.0704422 , 0.06938669, 0.06833318, 0.06729427,
       0.06627652, 0.06528361, 0.06431846, 0.06337486, 0.06243388,
       0.06149382, 0.06055262, 0.0596193 , 0.05869608, 0.0577806 ,
       0.05687203, 0.05597022, 0.05507641, 0.05419214, 0.05331872,
       0.05245768, 0.05161122, 0.05078244, 0.04997514, 0.04919295,
       0.0484383 , 0.04771196, 0.0470135 , 0.04634189, 0.04569584,
       0.04507405, 0.04447523, 0.04389827, 0.04334223, 0.04280618,
       0.0422892 , 0.04179037, 0.04130873, 0.04084326, 0.04039274,
       0.03995583, 0.03953114, 0.03911741, 0.0387135 , 0.03831831,
       0.03793068, 0.03754933, 0.03717281, 0.03679955, 0.03642779,
       0.03605564, 0.03568108, 0.03530207, 0.03491696, 0.03452625,
       0.03413593, 0.03375713, 0.03339667, 0.03305322, 0.03272585,
       0.03241645, 0.03212523, 0.03184994, 0.03158792, 0.03133706,
       0.03109577, 0.03086274, 0.03063697, 0.03041765, 0.03020413,
       0.02999582, 0.02979215, 0.02959259, 0.02939666, 0.02920392,
       0.02901399, 0.02882652, 0.0286412 , 0.02845773, 0.02827589,
       0.02809547, 0.02791633, 0.02773838, 0.02756159, 0.02738596,
       0.02721152, 0.02703833, 0.02686644, 0.0266959 , 0.02652672,
       0.02635893, 0.0261925 , 0.02602744, 0.02586375, 0.02570147,
       0.02554071, 0.02538164, 0.02522454, 0.02506979, 0.02491778,
       0.02476873, 0.02462224, 0.02447686, 0.02433042, 0.02418208,
       0.0240334 , 0.02388626, 0.02374131, 0.0235985 , 0.02345767,
       0.02331873, 0.02318168, 0.02304654, 0.02291343, 0.02278248,
       0.02265383, 0.02252759, 0.02240384, 0.0222826 , 0.02216387,
       0.02204758, 0.02193364, 0.02182192, 0.02171227, 0.02160449,
       0.0214984 , 0.02139375, 0.02129028, 0.02118766, 0.0210855 ,
       0.02098326, 0.02088019, 0.02077508, 0.02066593, 0.02054911,
       0.02041891, 0.02027251, 0.02012665, 0.0200047 , 0.01990199,
       0.01980242, 0.01969892, 0.01959328, 0.01948969, 0.01939036,
       0.01929493, 0.0192014 , 0.01910615, 0.01900187, 0.01888088,
       0.01875004, 0.01862485, 0.01851025, 0.01840367, 0.01830442,
       0.01821279, 0.0181275 , 0.0180467 , 0.01796905, 0.0178938 ,
       0.01782052, 0.01774901, 0.0176791 , 0.01761069, 0.01754371,
       0.01747807, 0.0174137 , 0.01735048, 0.01728827, 0.01722686,
       0.01716598, 0.01710532, 0.0170445 , 0.01698304, 0.01692042,
       0.01685598, 0.01678894, 0.01671834, 0.01664309, 0.01656207,
       0.01647456, 0.0163808 , 0.01628245, 0.01618223, 0.01608273,
       0.01598552, 0.01589127, 0.01580045, 0.0157139 , 0.01563278,
       0.01555802, 0.01548975, 0.01542707, 0.01536846, 0.01531253,
       0.01525856, 0.01520643, 0.01515633, 0.01510834, 0.01506234,
       0.01501817, 0.01497566, 0.01493476, 0.01489543, 0.01485764,
       0.01482132, 0.0147864 , 0.01475281, 0.01472047, 0.01468929,
       0.0146592 , 0.01463012, 0.01460194, 0.01457454, 0.01454776,
       0.01452142, 0.01449534, 0.01446933, 0.01444328, 0.01441709,
       0.01439076, 0.0143643 , 0.01433777, 0.01431125, 0.0142848 ,
       0.01425852, 0.01423247, 0.01420672, 0.01418133, 0.01415634,
       0.01413177, 0.01410763, 0.01408392, 0.0140606 , 0.01403763,
       0.01401493, 0.01399244, 0.01397011, 0.01394791, 0.01392585,
       0.01390394, 0.01388225, 0.01386083, 0.01383976, 0.0138191 ,
       0.01379887, 0.01377911, 0.01375982, 0.01374097, 0.01372253,
       0.01370445, 0.01368666, 0.01366909, 0.01365165, 0.01363428,
       0.01361691, 0.01359946, 0.01358188, 0.01356414, 0.01354621,
       0.01352808, 0.01350974, 0.0134912 , 0.01347248, 0.01345359,
       0.01343456, 0.01341541, 0.01339616, 0.01337685, 0.01335749,
       0.01333812, 0.01331875, 0.01329942, 0.01328014, 0.01326096,
       0.01324188, 0.01322293, 0.01320414, 0.01318553, 0.01316711,
       0.0131489 , 0.01313093, 0.01311319, 0.01309571, 0.0130785 ,
       0.01306156, 0.0130449 , 0.01302853, 0.01301246, 0.01299668,
       0.01298119, 0.012966  , 0.01295111, 0.01293652, 0.01292221,
       0.0129082 , 0.01289447, 0.01288102, 0.01286784, 0.01285493,
       0.01284229, 0.01282989, 0.01281775, 0.01280585, 0.01279419,
       0.01278275, 0.01277154, 0.01276054, 0.01274975, 0.01273916,
       0.01272876, 0.01271855, 0.01270853, 0.01269867, 0.01268899,
       0.01267947, 0.01267011, 0.0126609 , 0.01265183, 0.01264291,
       0.01263412, 0.01262546, 0.01261692, 0.0126085 , 0.01260019,
       0.01259199, 0.01258389, 0.01257589, 0.01256798, 0.01256016,
       0.01255242, 0.01254475, 0.01253715, 0.01252961, 0.01252214,
       0.01251471, 0.01250733, 0.01249998, 0.01249267, 0.01248538,
       0.01247811, 0.01247085, 0.01246359, 0.01245632, 0.01244904,
       0.01244173, 0.0124344 , 0.01242701, 0.01241958, 0.01241208,
       0.01240451, 0.01239685, 0.01238909, 0.01238122, 0.01237323,
       0.01236509, 0.0123568 , 0.01234833, 0.01233968, 0.01233081,
       0.01232172, 0.01231239, 0.01230279, 0.01229291, 0.01228274,
       0.01227227, 0.01226149, 0.0122504 , 0.01223901, 0.01222734,
       0.01221541, 0.01220325, 0.01219091, 0.01217843, 0.01216586,
       0.01215326, 0.01214066, 0.01212812, 0.0121157 , 0.01210342,
       0.01209134, 0.01207948, 0.01206788, 0.01205657, 0.01204555,
       0.01203485, 0.01202448, 0.01201443, 0.01200472, 0.01199535,
       0.01198629, 0.01197756, 0.01196914, 0.01196101, 0.01195318,
       0.01194562, 0.01193833, 0.0119313 , 0.0119245 , 0.01191793,
       0.01191158, 0.01190543, 0.01189948, 0.01189371, 0.01188812,
       0.01188269, 0.01187742, 0.0118723 , 0.01186731, 0.01186246,
       0.01185774, 0.01185313, 0.01184864, 0.01184425, 0.01183997,
       0.01183579, 0.01183169, 0.01182769, 0.01182377, 0.01181993,
       0.01181617, 0.01181249, 0.01180887, 0.01180532, 0.01180184,
       0.01179842, 0.01179506, 0.01179176, 0.01178851, 0.01178532,
       0.01178218, 0.01177909, 0.01177605, 0.01177305, 0.0117701 ,
       0.01176719, 0.01176433, 0.01176151, 0.01175873, 0.01175598,
       0.01175328, 0.01175061, 0.01174797, 0.01174537, 0.01174281,
       0.01174027, 0.01173777, 0.0117353 , 0.01173286, 0.01173045,
       0.01172806, 0.01172571, 0.01172338, 0.01172108, 0.01171881,
       0.01171656, 0.01171434, 0.01171214, 0.01170996, 0.01170781,
       0.01170568, 0.01170357, 0.01170149, 0.01169943, 0.01169738,
       0.01169536, 0.01169336, 0.01169138, 0.01168942, 0.01168747,
       0.01168555, 0.01168364, 0.01168176, 0.01167989, 0.01167803,
       0.0116762 , 0.01167438, 0.01167258, 0.01167079, 0.01166902,
       0.01166727, 0.01166553, 0.0116638 , 0.01166209, 0.0116604 ,
       0.01165872, 0.01165706, 0.0116554 , 0.01165377, 0.01165214,
       0.01165053, 0.01164894, 0.01164735, 0.01164578, 0.01164422,
       0.01164268, 0.01164114, 0.01163962, 0.01163811, 0.01163661,
       0.01163513, 0.01163365, 0.01163219, 0.01163074, 0.01162929,
       0.01162786, 0.01162645, 0.01162504, 0.01162364, 0.01162225,
       0.01162087, 0.0116195 , 0.01161815, 0.0116168 , 0.01161546,
       0.01161413])

    graphic.plot(trError, 'r--')#tr set
    graphic.plot(vlError, 'b')#vl set
    graphic.show()
    """
    """
    Test della double cross-validation.
    """
    """
    f = Sigmoidal(12)

    domains = [3, 3, 2, 3, 4, 2]
    columnSkip = [8]
    targetPos = 1
    datasetFileName = "monks-1.train"

    trainingSet = DataSet(datasetFileName, " ", ModeInput.ONE_OF_K_TR_INPUT, targetPos, domains, None, columnSkip)
    learnRates = [0.1]
    momRates = [0.5, 0.6]
    regRates = [0]
    ValMaxs = [0.75]
    HiddenUnitss = [4]
    OutputUnitss = [1]
    MaxIterss = [400]
    Tolerances = [0.0001]
    start = time.time()
    e = double_cross_validation(2, 3, 3, trainingSet.getInputs(), f, learnRates, momRates, regRates, ValMaxs, HiddenUnitss, OutputUnitss, MaxIterss, Tolerances, start, None,ModeLearn.BATCH)
    stop = time.time() 
    secdiff = stop - start
    print("Errore: " + str(e) + ", tempo: " + str(secdiff))
    """