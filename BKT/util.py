# encoding:utf-8

# Inference routinue
def update_mastery(mastery, learn_rate):
	return mastery + (1-mastery)*learn_rate

def compute_success_rate(slip, guess, mastery):
	return guess*(1-mastery) + (1-slip)*mastery

# Bayesian Knowledge Tracing Algorithm
def forward_update_mastery(mastery, slip, guess, learn_rate, Y):
	if Y ==1:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*guess/(guess+(1-slip-guess)*mastery)
	elif Y==0:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*(1-guess)/(1-guess-(1-slip-guess)*mastery)
	else:
		raise ValueError('Invalid response value.')
	return new_mastery
	
def generate_learning_curve(slip, guess, init_mastery, learn_rate, T):
	p=init_mastery
	lc = [compute_success_rate(guess, slip, p)]
	for t in range(1,T):
		p = update_mastery(p,learn_rate)
		lc.append(compute_success_rate(guess, slip, p))
	return lc
	
	