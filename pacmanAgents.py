# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these
# projects for
# educational purposes provided that (1) you do not distribute or
# publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at
#  UC Berkeley.
# The core projects and autograders were primarily created by John
# DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math


class RandomAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):
		# get all legal actions for pacman
		actions = state.getLegalPacmanActions()
		# returns random action from all the valide actions
		return actions[random.randint(0, len(actions) - 1)]


class RandomSequenceAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		self.actionList = [];
		for i in range(0, 10):
			self.actionList.append(Directions.STOP);
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):
		# get all legal actions for pacman
		possible = state.getAllPossibleActions();
		for i in range(0, len(self.actionList)):
			self.actionList[i] = possible[
				random.randint(0, len(possible) - 1)];
		tempState = state;
		for i in range(0, len(self.actionList)):
			if tempState.isWin() + tempState.isLose() == 0:
				tempState = tempState.generatePacmanSuccessor(
					self.actionList[i]);
			else:
				break;
		# returns random action from all the valid actions
		return self.actionList[0];


class GreedyAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):
		# get all legal actions for pacman
		legal = state.getLegalPacmanActions()
		# get all the successor state for these actions
		successors = [(state.generatePacmanSuccessor(action), action)
					  for action in legal]
		# evaluate the successor states using scoreEvaluation
		# heuristic
		scored = [(scoreEvaluation(state), action) for state, action
				  in successors]
		# get best choice
		bestScore = max(scored)[0]
		# get all actions that lead to the highest score
		bestActions = [pair[1] for pair in scored if
					   pair[0] == bestScore]
		# return random action from the list of the best actions
		return random.choice(bestActions)


class HillClimberAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		self.actionList = [];
		for i in range(0, 5):
			self.actionList.append(Directions.STOP);
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):
		self.root = state
		flag = False
		new_action = Directions.STOP
		possible = self.root.getAllPossibleActions()
		newActionList = self.newSequence()
		maximum_score = 0

		while True:
			current_state = state
			for i in range(0, len(newActionList)):
				if current_state.isWin() + current_state.isLose() \
						== 0:
					successor = \
						current_state.generatePacmanSuccessor(
						newActionList[i])
					if successor is None:
						flag is True
						break
					else:
						current_state = successor
				else:
					break
			if flag is True:
				break
			new_score = scoreEvaluation(current_state)
			if new_score > maximum_score:
				maximum_score = scoreEvaluation(current_state)
				new_action = newActionList[0]

			for i, action in enumerate(newActionList):
				r_test = random.uniform(0, 1)
				if r_test > 0.5:
					newActionList[i] = random.choice(possible)

		return new_action

	def newSequence(self):
		newList = []
		possible = self.root.getAllPossibleActions()
		for i in range(0, 5):
			newList.append(random.choice(possible))
		return newList


class GeneticAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):

		# TODO: write Genetic Algorithm instead of returning
		possible = state.getAllPossibleActions()

		def init_population():
			gene_pool = state.getAllPossibleActions()
			g_len = len(gene_pool)
			population = []
			for i in range(0, 8):
				chromosome = []
				for j in range(0, 5):
					chromosome.append(
						gene_pool[
							random.randint(0, g_len - 1)]
					)
				population.append(chromosome)
			return population

		def selection_chances(state, population):
			chromosome_score = []
			score = 0
			tempstate = state
			result = []
			for direction_list in population:
				score = 0
				flag = True
				tempstate = state
				for direction in direction_list:
					if tempstate.isWin() + tempstate.isLose() == 0:
						successor = \
							tempstate.generatePacmanSuccessor(
							direction
						)
						if successor == None:
							flag = False
							break;
						tempstate = successor
						score = score + scoreEvaluation(successor)
					else:
						break
				if flag == False:
					break;
				chromosome_score.append(direction_list)
				chromosome_score_list = (score, direction_list)
				result.append(chromosome_score_list)
			result.sort(reverse=True)
			if flag is False:
				return 0
			else:
				return result

		def rank_selection(sorted_chromosome_list):
			total_chromosomes = 8
			tot_sum = 1
			denominator = sum(range(1, total_chromosomes + 1))
			proportional_probability = [
				round((x + 1) * 1.0 / denominator, 2) for x in
				range(8)]
			proportional_probability.sort(reverse=True)
			x = 0.00
			iterative_proportional_probability = []
			for i in range(len(proportional_probability)):
				iterative_proportional_probability.append(
					x + proportional_probability[i])
				x = iterative_proportional_probability[i]

			limit = random.uniform(0, tot_sum)
			index_of_first_item_bigger_than = next(
				x[0] for x in
				enumerate(iterative_proportional_probability)
				if x[1] > limit
			)
			return \
			sorted_chromosome_list[
				index_of_first_item_bigger_than][1]

		# rank_selection(sorted_chromosome_list)
		def chromosomeCrossover(parent1, parent2):
			new_chromosome = []
			for i in range(0, 5):
				x = random.uniform(0, 1)
				if x < 0.5:
					new_chromosome.append(parent1[i])
				else:
					new_chromosome.append(parent2[i])

			return new_chromosome;

		def chromosomeMutate(new_population):
			for chromosome in new_population:
				r_num = random.randint(0, 1)
				if r_num <= 0.1:
					chromosome[random.randint(0, 4)] = possible[
						random.randint(0, len(possible) - 1)]
			return new_population

		def gen_algo():
			population = init_population()
			population_selection = selection_chances(state,
													 population)
			while True:
				if population_selection == 0:
					break;
				else:
					parent1 = rank_selection(population_selection)
					parent2 = rank_selection(population_selection)
					r_test = random.randint(0, 1)
					if r_test < 0.7:
						for i in range(0, 2):
							new_individual = chromosomeCrossover(
								parent1, parent2)
							for i in range(len(population)):
								if population[i] == parent1 or \
												population[
													i] == parent2:
									population[i] = new_individual
					new_population = chromosomeMutate(population)
					population_selection = selection_chances(state,
															 new_population)

					next_action = population_selection[1][1][0];
					return next_action

		next_action = gen_algo()
		return next_action


class MCTSAgent(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):
		# TODO: write MCTS Algorithm instead of returning
		# Directions.STOP
		self.root = state
        self.visits = {}
        self.parent = {}
        self.actions = {}
        self.reward = {}

        current = self.root
        self.reward[current] = 0
        self.parent[current] = None
        self.visits[current] = 0
        self.actions[current]= None

        flag = False
        while True:
                if flag== True: break
                current = self.root
                while(True):
                    flag1 = False
                    actions = current.getLegalPacmanActions()
                    UCT = 0
                    chosen_action = Directions.STOP
                    for action in actions:
                        x = current.generatePacmanSuccessor(action)
                        if (x == None):
                            break
                        if (x.isWin() + x.isLose()): break
                        if self.visits.get(x, 0) == 0:
                            self.parent[x] = current
                            self.actions[x] = action
                            self.visits[x] = 1
                            self.reward[x] = self.reward.get(x,0) + self.Rollout(x)
                            while self.parent.get(x,None)!= None:
                                self.visits[self.parent[x]] = self.visits.get(self.parent[x],0) + 1
                                self.reward[self.parent[x]] = self.reward.get(self.parent[x],0) + self.reward[x]
                                x = self.parent[x]
                            flag1 = True
                            break
                        else:
                            temp = self.UCT(current, x)
                            if temp > UCT:
                                UCT = temp,
                                chosen_action = action

                    if flag1 == True: break
                    current = current.generatePacmanSuccessor(chosen_action)
                    if current == None:
                        flag = True
                        break

        sorted_x = sorted(self.visits.items(), key=lambda value: value[1], reverse=True)
        print sorted_x,"sorted_x"
        node = Directions.STOP

        for index,w in enumerate(sorted_x):
            if index==1:
                node = w[0]
                break

        print self.actions[node]
        return self.actions[node]

    def UCT(self,current,state):
        value = self.reward[state]/float(self.visits[state]) + math.sqrt(2*math.log(self.visits[current])/float(self.visits[state]))
        #print value,"UCB"
        return value

    def Rollout(self,cur):
        current = cur
        for i in range(0,5):
            if (current.isLose() + current.isWin()!=0):
                return normalizedScoreEvaluation(self.root,current)
            else:
                action = random.choice(current.getAllPossibleActions())
                successor = current.generatePacmanSuccessor(action)
                if successor == None: break
                current = successor
        return normalizedScoreEvaluation(self.root,current)
