from enum import Enum
from math import exp,log
import math
import functools, sys
import numpy as np

"""
Classe che implementa la funzione di attivazione, con relativo calcolo della derivata prima.
"""
class ActivFunct(object):
    def __init__(self, f, derivative):
        self.f = f
        self.derivative = derivative      
    
    #Restituisce il valore della funzione di attivazione calcolata in x.
    def getf(self, x):
       return self.f(x)
        
    #Restituisce l'np.array delle derivate calcolate componente per componente
    def getDerivative(self, x:np.array):
        return self.derivative(x)

class Sigmoidal(ActivFunct):

    def __init__(self, alfa = 1):
        self.alfa = alfa

    def getf(self, x):
        power = -self.alfa*x
        return 1/(1+(np.exp(power)))

    def getDerivative(self, x:np.array):
        return self.alfa*self.getf(x)*(1- self.getf(x))

class SymmetricSigmoidal(Sigmoidal):
    def __init__(self, alfa = 1, beta = 2):
        self.alfa = alfa
        self.beta = beta
        self.sigm = Sigmoidal(self.alfa)

    def getf(self,x):
        return self.beta * self.sigm.getf(x) - self.beta/2
    
    def getDerivative(self,x:np.array):
        return self.beta * self.sigm.getDerivative(x)

class Identity(ActivFunct):
    def __init__(self):
        pass

    def getf(self, x):
        return x

    def getDerivative(self, x:np.array):
        return np.ones((len(x)))

class SoftPlus(ActivFunct):
    def __init__(self):
        pass
    
    def getf(self, x):
        return np.log(1+np.exp(x))

    def getDerivative(self, x):
        return np.exp(x)/(1+np.exp(x))


"""
f = Sigmoidal(0)

print(f.getf(1))
print(f.getDerivative(1))

i = Identity()

print(i.getf(10))
print(i.getDerivative(10))

"""
