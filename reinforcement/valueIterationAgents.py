# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            values = util.Counter()  # We use this cos apparently autograder needs this to work
            
            for state in mdp.getStates():
                if not mdp.isTerminal(state):

                    best_value = float("-inf")
                    
                    # Compute V(s) by picking the max Q(s, a) for each possible action
                    for action in mdp.getPossibleActions(state):
                        value = self.computeQValueFromValues(state, action) # Q(s, a)

                        if value > best_value:
                            best_value = value

                    values[state] = best_value

            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0

        # Next_states is a list of all the possible states following the current state, as well as the probability
        # that we will end up in each state after taking the action
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # We loop through each state and 
        for next_state in next_states:
            # We multiply the probability of reaching the next_state with
            # 1) the reward of taking the action from the current state and ending up in that particular next state
            # 2) the value of the next state multiplied by the discount
            value += next_state[1] * (self.mdp.getReward(state, action, next_state[0]) + self.discount * self.values[next_state[0]])

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None


        actions = self.mdp.getPossibleActions(state)

        # This list represents [best possible action, value of taking that action from the current state]
        # We set the best possible action to None first so that if there are no actions we just return None
        best = [None, float("-inf")]

        # For each action we compute the Q value of and we pick the action with the highest value
        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if value > best[1]:
                best = [action, value]

        return best[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
