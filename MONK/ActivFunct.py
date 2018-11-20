"""
Classe che implementa la funzione di attivazione, con relativo calcolo della derivata prima.
"""
class ActivFunct(object):

    #Costruttuore: inserisco come attributo la funzione di attivazione fissati i parametri aggiuntivi.
    #ACHTUNG: i parametri aggiuntivi vanno posti in testa alla lista!
    def __init__(self, f, param : list):
        from functools import partial
        from inspect import signature

        #len(signature(f).parameters)-1 restituisce il numero di parametri aggiuntivi di f.
        if len(param) == len(signature(f).parameters)-1:
            self.f = f
            for el in param:
                self.f = partial(f,el)
        else:
            raise ValueError ('ActivFun: number of parameters between f and list mismatch.')
    
    #Restituisce il valore della funzione di attivazione calcolata in x.
    def getf(self, x):
        return self.f(x)
    
    #Restituisce il valore della derivata prima (differenze finite, scipy.misc.derivative)
    #della funzione di attivazione calcolata in x.
    def getDerivative(self,x):
        from scipy.misc import derivative
        return derivative(self.f,x,dx=10e-10)

"""
def sigmoidal(a,x):
    from math import exp
    return 1/(1+exp(-a*x))

f = ActivFunct(sigmoidal,[1])
print(f.getf(0.000012334))
print(f.getDerivative(0.000012334))

# Atteso lancio di eccezione.
f = ActivFunct(sigmoidal,[1, 2, 3, 4])
"""
