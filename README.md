# pyMLC
a python implementation of mixture learning curve model

# Documentation
Mixture Learning Curve (MLC) model is proposed by [Matthew Streeter(2015)](http://www.educationaldatamining.org/EDM2015/proceedings/full45-52.pdf). The original paper is light on the derivation, which I supplemented in the document folder.

The canonical Bayesian Knowledge Tracing (BKT) can be viewed as a special case of the MLC. It imposes that the learning curve of each component is a step function with same upper and lower limit while the mixture density is govern by learning rate the prior. 

# Specification

The vanilla version of MLC imposes no structure on the learning curve. Therefore it also loses the ability to generate user specific learning curve parameter.


# Missing Data
It is likely that student's response is right truncated. For example, if the max practice
opportunity is 5, student's spell can practice 2 or 3 times. The last few obs
is thus missing.

If the missing data is not correlated with the learning curve (missing at
random), the estimator is consistent.


# (Un)identification of the learning curve
The mixed learning curve is not uniquely identified. The alogrithm fit 5 curves
and chosen the curvature with largest l2 norm. The intuition is that we want to
distinguish user with different learning speed.
