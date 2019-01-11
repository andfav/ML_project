from enum import Enum
from math import exp,log
import math
import functools, sys

"""
Classe che implementa la funzione di attivazione, con relativo calcolo della derivata prima.
"""
class ActivFunct(object):
    #Costruttuore: inserisco come attributo la funzione di attivazione fissati i parametri aggiuntivi.
    #ACHTUNG: i parametri aggiuntivi vanno posti in testa alla lista!
    def __init__(self, f, derivative):
        self.f = f
        self.derivative = derivative

        #len(signature(f).parameters)-1 restituisce il numero di parametri aggiuntivi di f.
        
    
    #Restituisce il valore della funzione di attivazione calcolata in x.
    def getf(self, x):
       return self.f(x)
        
    
    #Restituisce il valore della derivata prima (differenze finite, scipy.misc.derivative)
    #della funzione di attivazione calcolata in x.
    def getDerivative(self,x):
        return self.derivative(x)

class Sigmoidal(ActivFunct):
    def __init__(self, alfa = 1):
        self.alfa = alfa

    def getf(self, x):
        power = -self.alfa*x
        return 1/(1+(exp(power)))

    def getDerivative(self, x):
        return self.alfa*self.getf(x)*(1- self.getf(x))

class SymmetricSigmoidal(ActivFunct):
    def __init__(self, alfa = 1, beta = 2):
        self.alfa = alfa
        self.beta = beta
        self.sigm = Sigmoidal(self.alfa)

    def getf(self,x):
        return self.beta * self.sigm.getf(x) - self.beta/2
    
    def getDerivative(self,x):
        return self.beta * self.sigm.getDerivative(x)

class Identity(ActivFunct):
    def __init__(self):
        pass

    def getf(self, x):
        return x

    def getDerivative(self, x):
        return 1

class SoftPlus(ActivFunct):
    def __init__(self):
        pass
    
    def getf(self, x):
        return log(1+exp(x))

    def getDerivative(self, x):
        return exp(x)/(1+exp(x))


"""
f = Sigmoidal(0)

print(f.getf(1))
print(f.getDerivative(1))

i = Identity()

print(i.getf(10))
print(i.getDerivative(10))

"""
