# Learning Through Practices(LTP)

## 1.Hidden Markov Model

### 1.1 Bayesian Knowledge Tracing
The canonical Bayesian Knowledge Tracing (BKT) is the special case of left-right HMM with a single item where the number of latent states and that of observe states are both TWO.

The model can be fitted by EM algorithm or by MCMC algorithm.

It should be noticed that the parameters of BKT are uniquely identified if the specification is right.

### 1.2 General LTP model with response only

There are good reasons why the number of latent states or the number of observation states are more than two. For example, the zone of proximal development suggests the correct latent state is 3. For another example, partial credit can be roughly modeled as a three-state observation. Furthermore, rarely does the practice sequence consists of only ONE item. 

The general LTP model, estimated with MCMC, allows for multi-item and multi-state HMM model. 

The model imposes the "no-forgetting" constraints. 
```
P(X_t=m|X_{t-1}=n) = 0 if m<n 
```

The default model set the number of state of latent mastery (Mx) to 2. The number of state of the observed response (My) depends on the input data.

If My=2 and Mx=2, a default rank order is imposed according to BKT's tradition. Other than this specification, the user needs to specify his/her own rank order condition to prevent state switching.

#### 1.2.1 Data Input
The input has the format of (i,t,j,y), where

* i: user id, starts from 0 and continuous integer
* t: sequence id, starts from 0 and continuous integer
* j: item id, starts from 0 and continuous integer 
* y: discrete state of response. e.g, 0/1

#### 1.2.2 Usage

The default use where Mx=2
```python
from LTP import LTP_HMM_MCMC
mcmc_instance = LTP_HMM_MCMC()
est_param = mcmc_instance.estimate(input_data)
```

Change the number (default 4) and length(default 1000) of the markov chain
```python
est_param = mcmc_instance.estimate(input_data, max_iter=100, chain_num=1)
```



Add three states. Assume My=3, add rank order condition that P(Y=2|X=0)=P(Y=0|X=2) = 0
```python
zms = {'Y':[(0,2),(2,0)]} #(X,Y)
est_param = mcmc_instance.estimate(input_data, zero_mass_set = zms)
```

Further add constraints that transition is local, P(Xt=m|Xt-1=n)=0 if m-n>1
```python
zms = {'Y':[(0,2),(2,0)], #(X,Y)
		'X':[(0,2)] #(Xt-1,Xt)
	}
est_param = mcmc_instance.estimate(input_data, zero_mass_set = zms)
```



#### 1.2.3 Demo
For simulation, check the example at *demo/hmm_demo.py*



### 1.3 General LTP model with Different Spell Lengths and Effort Choice
Easter egg. Not sure it is ready for prime time.




## Mixture Learning Curve

Mixture Learning Curve (MLC) model is proposed by [Matthew Streeter(2015)](http://www.educationaldatamining.org/EDM2015/proceedings/full45-52.pdf). The original paper is light on the derivation, which I supplemented in the document folder.
BKT can be viewed as a special case of the MLC. It imposes that the learning curve of each component is a step function with same upper and lower limit while the mixture density is governed by learning rate the prior. 
The vanilla version of MLC imposes no structure on the learning curve. Therefore it also loses the ability to generate user-specific learning curve parameter.

### Homogeneous Item Parameters and Unidimensionality

Assume all practice opportunities are essentially the same.

Assume the latent ability is uni-dimensional.

### Different Spell Length
Have to assume missing at random. 

### (Un)identification of the learning curve

The mixed learning curve is not uniquely identified. The algorithm fit 5 curves and chosen the curvature with largest $\ell_2$ norm. 

The intuition is that we want to distinguish user with different learning speed.
