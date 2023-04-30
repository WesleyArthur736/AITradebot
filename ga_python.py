import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure

'''
1. The problem - optimization

suppose we have a function f(x) and we want to find 
min f(x) where x = (x_1, x_2, ..., x_n) is an element in R^n.
Let x* be the argmin to f(x). We want to find x*.

2. The general structure of an Evolutionary Algorithm:
	a. initialize the population
	b. select instances to reproduce based on their fitness
	c. reproduce these selected instances
	d. repeat steps b and c until some termination condition. 

3. The general structure of a Genetic Algorithm:
	a. initialize the population
	b. select instances (parents) and perform crossover to create 2 offspring. 
	c. mutate the offspring
	d. merge the main population and the mutated offspring.
	e. evaluate, sort and select the op performers. 
	f. repeat from step b, until some termination condition. 


Crossover:

An operator that accepts information from two parents and returns the information 
for two offspring. See single point, double point and uniform crossover example 
with a binary genome in video 4.

Mutation:

An operator which takes a single population instance and returns a mutated 
version of this solution. See video 5 on the simple equations for binary mutation. 

Parent Selection:

The simplest method of selecting parents for reproduction is random selection.
A second method is "tournament selection". Select parents from the group based upon 
fitness value (deterministic or probabilistic). A third method is "Roulette Wheel Selection":
basically represent the population fitness values as a pie chart and "spin" this chart to select. 

Merging, Sorting and Selecting:

We have the parent population "pop_t" of size n_pop and the mutated population "children_t" 
of size m. If we merge these two populations we get a population of size "n_pop + m". 

We sort the resultant population according to the objective function. Better members are on top
of the less fit members.

We then select "n_pop" members from the resultant population to form "pop_t+1". We do this by 
removing the "m" least fit solutions from the resultant population. See video 7 for details. 
'''


'''
Consider this simple sphere optimization problem:

We want to minimize f_sph(x) = \sum{i = 1}^{n}x_i where x_i^2 \in Reals.
the solution is obvious: x* = 0. 
'''



## An implementation of the "sphere" cost function:
def sphere(x):
	return sum(x ** 2)


## Problem definition
problem = structure() # a "ypstruct" structure that will essentially act as a "class" for the problem instance
problem.costfunc = sphere # the cost function 
problem.nvar = 5 # the number of variables
problem.varmin = -10 # the lower bound on variable values
problem.varmax = 10 # the upper bound on variable values


## GA parameters
params = structure()
params.maxit = 200 # the maximum number of iterations to run the algorithm
params.npop = 50 # the population size 
params.beta = 1
params.pc = 1 # a number that determines the proportion of children to the main population (1 means an equal split of offspring to main population)
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

## Define the genetic algorithm - see section 4 of the course:
def run_genetic_algorithm(problem, params):
	
	# problem instance information:
	costfunc = problem.costfunc
	nvar = problem.nvar
	varmin = problem.varmin
	varmax = problem.varmax

	# Parameters
	maxit = params.maxit
	npop = params.npop
	beta = params.beta
	pc = params.pc
	nc = int(np.round((pc * npop) / 2) * 2) # the number of children (this number is always even)
	gamma = params.gamma
	mu = params.mu
	sigma = params.sigma

	# create the empty individual template:
	empty_individual = structure()
	empty_individual.position = None
	empty_individual.cost = None

	# best solution so far:
	# deepcopy makes sure that changes to empty_individual do not effect bestsol, and vice versa
	bestsol = empty_individual.deepcopy()
	bestsol.cost = np.inf # the cost of the original bestsol is infinite (since this is a minimization problem)

	# create the initial population:
	pop = empty_individual.repeat(npop) # an array of "npop", empty individuals
	for i in range(0, npop):
		# create "nvar" uniformly distributed random numbers from "varmin" to "varmax"
		# this will initialize the initial population positions randomly:
		pop[i].position = np.random.uniform(low = varmin, high = varmax, size = nvar)
		pop[i].cost = costfunc(pop[i].position)

		# if pop[i] is better than anything found so far, then pop[i] is the new bestsol
		if pop[i].cost < bestsol.cost:
			bestcol = pop[i].deepcopy()

	# best cost of iterations - at the end of all iterations:
	bestcost = np.empty(maxit)

	# MAIN LOOP:
	for iteration in range(0, maxit):

		# costs = np.array([x.cost for x in pop]) # array of cost values of each individual in the population
		# avg_cost = np.mean(costs)
		# if avg_cost != 0:
		# 	costs = costs / avg_cost

		# # the selection probabilities:
		# probs = np.exp(-beta * costs)

		probs = get_selection_probabilities(pop, beta)

		# the population of offspring
		offspring_population = []
		for k in range(nc//2): # we have two offspring per crossover operation
			
			# # Select Parents - randomly in this case
			# parent_1, parent_2 = random_selection(pop, npop)

			# Select Parents - Roulette Wheel Selection:
			parent_1 = pop[roulette_wheel_selection(probs)]
			parent_2 = pop[roulette_wheel_selection(probs)]

			# Perform Crossover:
			child_1, child_2 = crossover(parent_1, parent_2)

			# Perform Mutation:
			child_1 = mutate(child_1, mu, sigma)
			child_2 = mutate(child_2, mu, sigma)

			# Apply Bounds - to make sure solutions are in valid range:
			apply_bound(child_1, varmin, varmax)
			apply_bound(child_2, varmin, varmax)

			# Evaluate The Offspring:
			child_1.cost = costfunc(child_1.position)
			child_2.cost = costfunc(child_2.position)

			if child_1.cost < bestsol.cost:
				bestsol = child_1.deepcopy()

			if child_2.cost < bestsol.cost:
				bestsol = child_2.deepcopy()

			# Add Offspring to The Population:
			offspring_population.append(child_1)
			offspring_population.append(child_2)

		# Merge, Sort and Select:
		pop += offspring_population # merge the offspring with the rest of the population
		pop = sorted(pop, key = lambda x: x.cost) # sort the population by fitness value 
		pop = pop[0:npop] # select the fittest "npop" members of the population

		# Store Best Cost:
		bestcost[iteration] = bestsol.cost

		# Print Iteration Information:
		print(f"Iteration: {iteration}, Best Cost: {bestcost[iteration]}")

	# output:
	out = structure()
	out.pop = pop
	out.bestsol = bestsol
	out.bestcost = bestcost

	return out

def get_selection_probabilities(pop, beta):
	costs = np.array([x.cost for x in pop]) # array of cost values of each individual in the population
	avg_cost = np.mean(costs)
	if avg_cost != 0:
		costs = costs / avg_cost

	# the selection probabilities:
	probs = np.exp(-beta * costs)

	return probs

def random_selection(pop, npop):
	# Select Parents - randomly in this case
	# we'll select two random indices from 0 to npop - 1
	rand_perm_npop = np.random.permutation(npop) # a random permutation of the population indices
	parent_1 = pop[rand_perm_npop[0]] # select parent 1 by choosing the instances from the population at the random index defined from above
	parent_2 = pop[rand_perm_npop[1]] # select parent 2 by choosing the instances from the population at the random index defined from above

	return parent_1, parent_2

def roulette_wheel_selection(p):
	'''
	p: an array of probabilities
	'''
	c = np.cumsum(p)
	r = sum(p) * np.random.rand()
	indicies = np.argwhere(r <= c)

	return indicies[0][0]


def crossover(p1, p2, gamma = 0.1):
	# uniform crossover
	c1 = p1.deepcopy()
	c2 = p2.deepcopy()

	alpha = np.random.uniform(
		-gamma,
		1 + gamma,
		*c1.position.shape
	)

	# randomized, linear combination of parents genome:
	c1.position = alpha * p1.position + (1 - alpha) * p2.position
	c2.position = alpha * p2.position + (1 - alpha) * p1.position

	return c1, c2

def mutate(x, mu, sigma):
	'''
	We will mutate the solution "x" at a rate of "mu" determined by a step size of "sigma"
	'''
	y = x.deepcopy() # y will be the mutated solution
	flag_arr = np.random.rand(*x.position.shape) <= mu # use * because x.position.shape is a tuple
	indicies = np.argwhere(flag_arr) # return the indicies where the flag is true

	y.position[indicies] = x.position[indicies] + sigma * np.random.randn(*indicies.shape)

	return y 

def apply_bound(x, varmin, varmax):
	x.position = np.maximum(x.position, varmin) # output of this is always >= varmin 
	x.position = np.minimum(x.position, varmax) # output of this is always <= varmax 

## Run GA
output = run_genetic_algorithm(problem, params)
# print(f"output.pop:\n{output.pop}")


## Results
# plt.plot(output.bestcost)
plt.semilogy(output.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("genetic Algorithm")
plt.grid(True)
plt.show()
