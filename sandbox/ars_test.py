# adaptive rejection sampler
import numpy as np

Xs = [-1,1]
J = len(Xs)
#hx
def h(x, mu=0,sig=1):
	# log kernal of the normal distribution
	return -(x-mu)**2/(2*sig**2)
def hprime(x, mu=0, sig=1):
	return -2(x-mu)/(2*sig**2)
	
#ux
Zs = []
for j in range(J):
	x0 = Xs[j]
	x1 = Xs[j+1]
	zj = (h(x1)-h(x0)- x1*hprime(x1)+x0*hprime(x0))/(hprime(x0)-hprime(x1))
	Zs.append(zj)
	

#lx

#sx