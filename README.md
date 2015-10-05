# pyMLC
a python implementation of mixture learning curve model

# Documentation
Mixture Learning Curve (MLC) model is proposed by Matthew Streeter[http://www.educationaldatamining.org/EDM2015/proceedings/full45-52.pdf] (2015). The original paper is light on the derivation, which I supplemented in the document folder.

The canonical Bayesian Knowledge Tracing (BKT) can be viewed as a special case of the MLC. It imposes that the learning curve of each component is a step function with same upper and lower limit while the mixture density is govern by learning rate the prior. 

# Specification

The vanilla version of MLC imposes no structure on the learning curve. Therefore it also loses the ability to generate user specific learning curve parameter.