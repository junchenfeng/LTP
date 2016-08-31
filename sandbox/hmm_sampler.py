import numpy as np
from scipy.stats import norm
import ipdb


A = np.array([[0.6,0.3,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])

us = np.array([-2,0,2])

sigma = 1

N = 1000


X = []
Y = []
for n in range(N):
	# sample initial from the stable distribution
	if n ==0:
		x = np.random.choice(3,p=np.array([0.2,0.6,0.2]))
	else:
		x = np.random.choice(3,p=A[X[n-1],:])
	
	# now sample Y
	y = np.random.randn()*sigma + us[x]
	
	X.append(int(x))
	Y.append(y)

	

rho = np.random.dirichlet((1,1,1))
As = np.zeros((3,3))
mus = np.zeros((3,))
sigma = 0

# prior dirichlet (1,1,...,1)
for i in range(3):
	As[i,:] = np.random.dirichlet((1,1,1))

# normal prior for u: eta = (min(Y)+max(Y))/2, tao = max(y)-min(y)
eta = (min(Y)+max(Y))/2
R = max(Y)-min(Y)
tao = 1/R**2
for i in range(3):
	mus[i] = np.random.randn()*R + eta
	
# normal prior for sigma Gamma(2, \beta) where \beta ~ Gamma(g=0.2, h=10/R^2)
g = 0.2
h = 10/R**2
beta = np.random.gamma(g, 1/h)
a = 2
sigma = np.random.gamma(a, 1/beta)

def log_exp_sum(llk_vec):
	llk_max = max(llk_vec)
	logsum = llk_max + np.log(np.exp(llk_vec-llk_max).sum())
	return logsum

	
T = 500
parameter_chain = np.zeros((T,3))
	
for k in range(T):
	# forward backward 
	a_vec = np.zeros((3,N))
	for t in range(N):
		if t==0:
			for i in range(3):
				a_vec[i,t] = np.log(rho[i]*norm.pdf(Y[t], mus[i], sigma))
		else:
			for i in range(3):
				llk_vec = np.array([np.log(As[j,i]) + np.log(norm.pdf(Y[t], mus[i], sigma)) + a_vec[j,t-1] for j in range(3)])
				a_vec[i,t] = log_exp_sum(llk_vec)			

	b_vec = np.zeros((3,N))
	for t in range(N-1,-1,-1):
		if t == N-1:
			b_vec[i,t] = 0  # log(1)
		else:
			for i in range(3):
				llk_vec = [np.log(As[i,j]) + np.log(norm.pdf(Y[t+1],mus[j],sigma)) +  b_vec[j,t+1] for j in range(3)]
				b_vec[i,t] = log_exp_sum(llk_vec)

	X_samples = []
	for t in range(N):
		weight = np.zeros((3,))
		if t == 0:	
			for i in range(3):
				# P(X_1=j) propto rho_j * pN(y_1:mu_j,sigma^2) * p_theta(y_2:N|X_1=j)
				weight[i] = np.log(rho[i]) + np.log(norm.pdf(Y[t], mus[i], sigma)) + b_vec[i,t]
		else:
			for i in range(3):
				# P(X_k=j|X_k-1 = i) propto a_ij * pN(y_k:mu_j,sigma^2) * p_theta(y_k+1:N|X_k=j)
				weight[i] = np.log(As[X_samples[t-1],i]) + np.log(norm.pdf(Y[t], mus[i], sigma)) + b_vec[i,t]
		# reweight
		weight = weight - log_exp_sum(weight)
		x = int(np.random.choice(3, p=np.exp(weight)))
		
		X_samples.append(x)

	# Update the posterior probability 

	transit_N = np.zeros((3,3))
	for t in range(1,N):
		transit_N[X_samples[t-1], X_samples[t]] += 1
		
	# Gamma(aij) = D(ni1+1,ni2+1, ..nij+1)
	for i in range(3):
		As[i,:] = np.random.dirichlet((1+transit_N[i,0], 1+transit_N[i,1], 1+transit_N[i,2]))

	# mu_i = N( (Si+tao*eta*sigma^2)/(ni+tao*sigma^2), sigma^2/(ni+tao*sigma^2) ), Si = sum_(X=i)(y), ni=sum(X=i)	
	X_N = np.zeros((3,))
	S_N = np.zeros((3,))
	for i in range(3):
		X_N[i] = len([x for x in X_samples if x ==i])
		S_N[i] = sum([Y[t] for t in range(N) if X_samples[t]==i])
		mu_posterior = (S_N[i]+tao*eta*sigma**2)/(X_N[i]+tao*sigma**2)
		std_posterior = (sigma**2/(X_N[i]+tao*sigma**2))**(0.5)
		mus[i] = np.random.randn() * std_posterior + mu_posterior

	# posterior 
	# sigma^-2 ~ G(a+0.5*n, \beta+0.5*(sum((y_k-mu_{Xk})^2)) )
	b_posterior = beta+0.5*sum([(Y[t]-mus[X_samples[t]])**2 for t in range(N)])
	sigma = np.random.gamma(a+0.5*N, 1/b_posterior)
	# beta = Gamma(g+a,h+sigma^-2)
	beta = np.random.gamma(g+a, 1/(h+sigma**(-2)))
	
	
	parameter_chain[k,:] = mus

ipdb.set_trace()
  