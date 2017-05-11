import numpy as np
import random
import math
from tqdm import tqdm
from collections import defaultdict
"""
class ARS
Created on Fri Mar 27 06:46:02 2015

@author: John Grenall (original author), Alberto Lumbreras (reviewed and cleaned)
@modified by Junchen Feng
"""

#TODO: speed up by use X,D patterns. pattern is at most (T+1)*D, which is a fraction of the original data

	
# construct the H function

def loglikelihood(Lambda, betas, x, d):
	xb = np.dot(x, betas)
	return (1-d)*np.log(1-Lambda*np.exp(xb)) + d*(np.log(Lambda) + xb)

def prime_llk_lambda(Lambda, betas, x, d):
	e_xb = np.exp(-np.dot(x, betas))
	return -(1-d)/(e_xb-Lambda) + d/Lambda
	
# construct the H prime function
def prime_llk_beta(Lambda, betas, x, d, j):
	exb = np.exp(np.dot(x, betas))
	return (d-Lambda*exb)/(1-Lambda*exb)*x[j]
'''
# Unit test
x = np.array([1,0])
epsi=0.00001

d = 1
Lambda = 0.2
betas = np.array([np.log(1.5), np.log(1.1)])
betas1 = np.array([np.log(1.5)+epsi, np.log(1.1)])

lprime = prime_llk_lambda(Lambda,betas,x,d)
lprime_sim = (loglikelihood(Lambda+epsi,betas,x,d)-loglikelihood(Lambda,betas,x,d))/epsi
print(lprime, lprime_sim)

b1prime = prime_llk_beta(Lambda,betas,x,d,0)
b1prime_sim = (loglikelihood(Lambda,betas1,x,d)-loglikelihood(Lambda,betas,x,d))/epsi
print(b1prime, b1prime_sim)

d = 0
Lambda = 0.2
betas = np.array([np.log(1.5), np.log(1.1)])
betas1 = np.array([np.log(1.5)+epsi, np.log(1.1)])

lprime = prime_llk_lambda(Lambda,betas,x,d)
lprime_sim = (loglikelihood(Lambda+epsi,betas,x,d)-loglikelihood(Lambda,betas,x,d))/epsi
print(lprime, lprime_sim)

b1prime = prime_llk_beta(Lambda,betas,x,d,0)
b1prime_sim = (loglikelihood(Lambda,betas1,x,d)-loglikelihood(Lambda,betas,x,d))/epsi
print(b1prime, b1prime_sim)
'''



	
	
# sample

def tot_llk(Lambda,betas,state_dict):
	llks = 0
	for obs_key, cnt in state_dict.items():
		Ds,Xs = obs_key.split('-')
		D = int(Ds)
		X = [int(x) for x in Xs.split('|')]
		llks += loglikelihood(Lambda,betas,X,D)*cnt
	return llks

def prime_tot_llk_lambda(Lambda,betas,state_dict):
	llk_primes = 0
	for obs_key, cnt in state_dict.items():
		Ds,Xs = obs_key.split('-')
		D = int(Ds)
		X = [int(x) for x in Xs.split('|')]
		llk_primes += prime_llk_lambda(Lambda,betas,X,D)*cnt
	return llk_primes

def prime_tot_llk_beta(Lambda,betas,state_dict,j):
	llk_primes_beta = 0
	for obs_key, cnt in state_dict.items():
		Ds,Xs = obs_key.split('-')
		D = int(Ds)
		X = [int(x) for x in Xs.split('|')]
		llk_primes_beta += prime_llk_beta(Lambda,betas,X,D,j)*cnt	
	return llk_primes_beta	

	
	

class ARS():
	'''
	This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
	Where possible, naming convention has been borrowed from this paper.
	The PDF must be log-concave.

	Currently does not exploit lower hull described in paper- which is fine for drawing
	only small amount of samples at a time.
	'''
	
	def __init__(self, f, fprima, xi=[-4,1,4], lb=-np.Inf, ub=np.Inf, use_lower=False, ns=50, **fargs):
		'''
		initialize the upper (and if needed lower) hulls with the specified params
		
		Parameters
		==========
		f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
		   density we want to sample from
		fprima:	 d/du log(f(u,...))
		xi: ordered vector of starting points in wich log(f(u,...) is defined
			to initialize the hulls
		D: domain limits
		use_lower: True means the lower sqeezing will be used; which is more efficient
				   for drawing large numbers of samples
		
		
		lb: lower bound of the domain
		ub: upper bound of the domain
		ns: maximum number of points defining the hulls
		fargs: arguments for f and fprima
		'''
		
		self.lb = lb
		self.ub = ub
		self.f = f
		self.fprima = fprima
		self.fargs = fargs
		
		#set limit on how many points to maintain on hull
		self.ns = 50
		self.x = np.array(xi) # initialize x, the vector of absicassae at which the function h has been evaluated
		self.h = np.array([self.f(x, **self.fargs) for x in self.x])
		self.hprime = np.array([self.fprima(x, **self.fargs) for x in self.x])
		
		#Avoid under/overflow errors. the envelope and pdf are only
		# proporitional to the true pdf, so can choose any constant of proportionality.
		self.offset = np.amax(self.h)
		self.h = self.h-self.offset 
		
		# Derivative at first point in xi must be > 0
		# Derivative at last point in xi must be < 0
		if not(self.hprime[0] > 0): 
			#print (self.hprime)			 
			raise IOError('initial anchor points must span mode of PDF')
		if not(self.hprime[-1] < 0):
			#print (self.hprime) 
			raise IOError('initial anchor points must span mode of PDF')
		self.insert() 

		
	def draw(self, N):
		'''
		Draw N samples and update upper and lower hulls accordingly
		'''
		samples = np.zeros(N)
		n=0
		while n < N:
			[xt,i] = self.sampleUpper()
			# TODO (John): Should perform squeezing test here but not yet implemented 
			ht = self.f(xt, **self.fargs)
			hprimet = self.fprima(xt, **self.fargs)
			ht = ht - self.offset
			#ut = np.amin(self.hprime*(xt-x) + self.h);
			ut = self.h[i] + (xt-self.x[i])*self.hprime[i]

			# Accept sample? - Currently don't use lower
			u = random.random()
			if u < np.exp(ht-ut):
				samples[n] = xt
				n +=1

			# Update hull with new function evaluations
			if self.u.__len__() < self.ns:
				self.insert([xt],[ht],[hprimet])

		return samples

	
	def insert(self,xnew=[],hnew=[],hprimenew=[]):
		'''
		Update hulls with new point(s) if none given, just recalculate hull from existing x,h,hprime
		'''
		if xnew.__len__() > 0:
			x = np.hstack([self.x,xnew])
			idx = np.argsort(x)
			self.x = x[idx]
			self.h = np.hstack([self.h, hnew])[idx]
			self.hprime = np.hstack([self.hprime, hprimenew])[idx]

		self.z = np.zeros(self.x.__len__()+1)
		
		# This is the formula explicitly stated in Gilks. 
		# Requires 7(N-1) computations 
		# Following line requires 6(N-1)
		# self.z[1:-1] = (np.diff(self.h) + self.x[:-1]*self.hprime[:-1] - self.x[1:]*self.hprime[1:]) / -np.diff(self.hprime); 
		self.z[1:-1] = (np.diff(self.h) - np.diff(self.x*self.hprime))/-np.diff(self.hprime) 

		self.z[0] = self.lb; self.z[-1] = self.ub
		N = self.h.__len__()
		idx_list = [0]+list(range(N))
		self.u = self.hprime[idx_list]*(self.z-self.x[idx_list]) + self.h[idx_list]
		# check if any of the exp(u) is inf
		if any([math.isnan(np.exp(u)) for u in self.u]):
			# inf encountered, means poor initial value, debug by print x and u
			#print(self.x)
			#print(self.u)
			raise Exception('X is poorly chosen.')
		else:
			self.s = np.hstack([0,np.cumsum(np.diff(np.exp(self.u))/self.hprime)])
			self.cu = self.s[-1]
			if math.isnan(self.cu) or self.cu<=0.0: 
				raise Exception('Invalid sample density.')


	def sampleUpper(self):
		'''
		Return a single value randomly sampled from the upper hull and index of segment
		'''
		u = random.random()
		
		# Find the largest z such that sc(z) < u
		i = np.nonzero(self.s/self.cu < u)[0][-1] 

		# Figure out x from inverse cdf in relevant sector
		xt = self.x[i] + (-self.h[i] + np.log(self.hprime[i]*(self.cu*u - self.s[i]) + 
		np.exp(self.u[i]))) / self.hprime[i]

		return [xt,i]
"""
	def plotHull(self):
		'''
		Plot the piecewise linear hull using matplotlib
		'''
		xpoints = self.z
		#ypoints = np.hstack([0,np.diff(self.z)*self.hprime])
		ypoints = np.exp(self.u) 
		plt.plot(xpoints,ypoints)
		plt.show()
		for i in range(1,self.z.__len__()):
			x1 = self.z[i]
			y1 = 0
			x2 = self.z[i+1]
			y2 = self.z[i+1]-self.z[i] * hprime[i]
"""
class ars_sampler(object):
	def __init__(self, Lambda, betas):
		self.Lambda = Lambda
		self.betas = betas

	def load(self,X,D):
		# read in the data
		self.X = X
		self.D = D
		self.N = self.D.shape[0]
		
		# collapse states here
		self.state_dict = defaultdict(int)
		for i in range(self.N):
			obs_key = str(self.D[i]) + '-' + '|'.join([str(x) for x in self.X[i,:]])
			self.state_dict[obs_key] += 1
	
	def sample_lambda(self,n=5):
	
		def f(x):
			return tot_llk(x, self.betas, self.state_dict)
			
			
		def fprima(x):
			return prime_tot_llk_lambda(x, self.betas, self.state_dict)
		bnds=[]
		for obs_key in self.state_dict.keys():
			X = [int(x) for x in obs_key.split('-')[1].split('|')]
			bnds.append(np.exp(-np.dot(X,self.betas)))
		bnd = min(bnds)
		
		is_fail = 0
		#TODO: better first guess
		alternative_low_guess = [0.05, 0.01]
		alternative_high_guess = [bnd-0.05, bnd-0.01, bnd-0.001]
		# check if the default mode makes sense
		guess_low = 0.1
		guess_high = min(bnd-0.1, 0.6)
		if fprima(guess_low)*fprima(guess_high)<0:
			try:
				ars = ARS(f, fprima, xi = [guess_low, (guess_low+guess_high)/2, guess_high], lb=0.01, ub=bnd)
			except:
				is_fail = 1
				print('Lambda not drew.')
		else:
			# check which side needs to be relaxed
			is_legit = 0
			if fprima(guess_low)<0:
				for gl in alternative_low_guess:
					if fprima(gl)>0:
						guess_low = gl
						is_legit = 1
						break
			if fprima(guess_high)>0:
				for gh in alternative_high_guess:
					if (gh<bnd) and fprima(gh)<0:
						guess_high = gh
						is_legit = 1
						break
			if is_legit:
				ars = ARS(f, fprima, xi = [guess_low, (guess_low+guess_high)/2, guess_high], lb=0.01, ub=bnd)
			else:
				is_fail = 1
				print('Lambda not drew.')

		if not is_fail:
			samples = ars.draw(n)
		else:
			samples = [min(self.Lambda,bnd-0.01)]

		return samples
	
	def sample_beta(self,k,n=5):
		def f(x):
			betas = np.copy(self.betas)
			betas[k] = x		
			return tot_llk(self.Lambda, betas, self.state_dict)
			
			
		def fprima(x):
			betas = np.copy(self.betas)
			betas[k] = x
			return prime_tot_llk_beta(self.Lambda, betas, self.state_dict, k)
		
		# only consider Xj!=0, it needs to be smaller than (-log(lambda)-X!=jb!=j)/Xj
		# also assume Xj>0 for now, otherwise needs to specify lower bnds by max((-log(lambda)-X!=jb!=j)/Xj)
		bnds=[]
		for obs_key in self.state_dict.keys():
			X = [int(x) for x in obs_key.split('-')[1].split('|')]
			if X[k]!=0:
				bnds.append( (-np.log(self.Lambda)-(np.dot(X,self.betas)-X[k]*self.betas[k]))/X[k] )
		bnd = min(bnds)
		
		# check input validity
		guess_low = min(-0.3, self.betas[k]-0.1)
		guess_high = bnd-0.1
		alternative_low_guess = [-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
		alternative_high_guess = [bnd-0.05, bnd-0.01,bnd-0.001]
		is_fail = 0
		if fprima(guess_low)*fprima(guess_high)<0:
			try:
				ars = ARS(f, fprima, xi = [guess_low, (guess_low+guess_high)/2, guess_high], lb=-1, ub=bnd)
			except:
				is_fail = 1
				print('Beta not drew.')
		else:		
			is_legit = 0
			if fprima(guess_low)<0:
				for gl in alternative_low_guess:
					if fprima(gl)>0:
						guess_low = gl
						is_legit = 1
						break
			if fprima(guess_high)>0:
				for gh in alternative_high_guess:
					if (gh<bnd) and fprima(gh)<0:
						guess_high = gh
						is_legit = 1
						break		
		
			if guess_low>guess_high:
				raise Exception('Wrong initial value for beta %d!'%k)	
			if is_legit:
				ars = ARS(f, fprima, xi = [guess_low, (guess_low+guess_high)/2, guess_high], lb=-1, ub=bnd)
			else:
				is_fail = 1
				print('Beta not drew.')
				
		if not is_fail:
			samples = ars.draw(n)
		else:
			samples = [min(self.betas[k],bnd-0.001)]

		return samples
	
	def mcmc(self):
		# initialize lambda, beta
		# TODO: The value cannot be totally random
		
		# iterative draw from the margins
		params = []
		for iter in tqdm(range(1000)):
			self.Lambda = self.sample_lambda()[-1]
			for k in range(3):
				val = self.sample_beta(k)[-1]
				self.betas[k] = val
			params.append([self.Lambda]+self.betas)
		return params
		
if __name__=='__main__':
	import os			  
	proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	file_path = proj_dir+'/data/bkt/test/single_sim_x_1.txt'
	X = []
	D = []

	with open(file_path) as in_f0:
		for line in in_f0:
			i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
			if int(is_a_s) == 1 and int(i_s)<1000:
				X.append( (int(t_s), int(x_s), int(x_s)*int(t_s)) )
				D.append(int(is_e_s))
				
	# calculate the empirical ratio
	h_cnt = np.zeros((5,2))
	s_cnt = np.zeros((5,2))
	for i in range(len(D)):
		t,x,xt=X[i]
		d=D[i]
		s_cnt[t,x]+=1
		h_cnt[t,x]+=d
	hrates = h_cnt/s_cnt
			

	X = np.array(X, dtype=np.int)
	D = np.array(D, dtype=np.int)
		
	test_obj = ars_sampler(0.1, [-0.01,0.01,0.01])
	test_obj.load(X,D)
	#test_obj.sample_lambda()
	#test_obj.sample_beta(1)
	params = test_obj.mcmc()
