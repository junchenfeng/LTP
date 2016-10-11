# Learning Through Practices(LTP)

## 1.Hidden Markov Model

### 1.1 Bayesian Knowledge Tracing
The canonical Bayesian Knowledge Tracing (BKT) is the special case of left-right HMM with a single item where the number of latent states and that of observe states are both TWO.

The model can be fitted by EM algorithm or by MCMC algorithm.

It should be noticed that the parameters of BKT are uniquely identified if the specification is right.

### 1.2 General LTP model with response only

There are good reasons why the number of latent states or the number of observation states are more than two. For example, the zone of proximal development suggests the correct latent state is 3. For another example, partial credit can be roughly modeled as a three-state observation. Furthermore, rarely does the practice sequence consists of only ONE item. 

The general LTP model, estimated with MCMC, allows for multi-item and multi-state HMM model. Unfortunately, the model is in general not uniquely identified. The package now allows for only 2 specifications.

* Mx = My = 2

similar to BKT.

* Mx=My = 3, 

The observation matrix restricts P(Y=2|X=0)=P(Y=0|X=2) = 0. 

The state tranisiton matrix is only admits diagonal and first off-diagonal entry on the upper-right matrix.  P(Xt=i|Xt-1=i)!=0 P(Xt=i|Xt-1=i-1) != 0. All other entries are 0.

#### 1.2.1 Data Input
The input has the format of (i,t,j,y), where

* i: user id, starts from 0 and continuous integer
* t: sequence id, starts from 0 and continuous integer
* j: item id, starts from 0 and continuous integer 
* y: discrete state of response. e.g, 0/1

#### 1.2.2 Usage

```python
from LTP import LTP_HMM_MCMC
mcmc_instance = LTP_HMM_MCMC()
est_param = mcmc_instance.estimate(input_data)
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