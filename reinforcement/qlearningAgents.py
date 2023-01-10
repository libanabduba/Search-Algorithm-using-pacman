# qlearningAgents.py
# ------------------
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


from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter()  # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Return the q value that has been stored
        if state in self.qvalues:
            return self.qvalues[state][action]

        # Return 0.0 if we have never seen a state
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        # If there are no possible actions, return 0.0
        if len(actions) == 0:
            return 0.0

        # Else return the maximum qvalue
        return max(self.getQValue(state, action) for action in actions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None  # Can't return an action if there are no actions to learn
        else:
            # Store the qvalues of taking each action from the state in a Counter
            values = util.Counter()
            for action in actions:
                values[action] = self.getQValue(state, action)

        # Return the action with the highest qvalue
        return values.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None  # Can't pick action if there are no actions to pick
        elif util.flipCoin(self.epsilon):  # Pick a random action if the probability is hit
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)  # Else pick the action with the highest value

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # The value of this state based on the reward for entering the state and the value of the next state
        sample1 = reward + self.discount * self.computeValueFromQValues(nextState)

        # If we have encountered this state before update the qvalue of taking the action in this state
        if state in self.qvalues:
            oldqvalue = self.qvalues[state][action]
            self.qvalues[state][action] = oldqvalue + self.alpha * (sample1 - oldqvalue)

        # Initialise a new qvalue counter for the state
        # and set the qvalue of taking the action in this state to the sample1 * alpha
        else:
            self.qvalues[state] = util.Counter()
            self.qvalues[state][action] = self.alpha * sample1

        return

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Return the product of the features and the weights
        return self.featExtractor.getFeatures(state, action) * self.weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # The value of this state based on the reward for entering the state and the value of the next state
        sample1 = reward + self.discount * self.getValue(nextState)

        # Difference between current qvalue estimate and true reward + estimate of Q from this state onwards
        difference = sample1 - self.getQValue(state, action)

        # Update the each weight for each feature based on the difference
        features = self.featExtractor.getFeatures(state, action)
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np


class NeuralNetQAgent(PacmanQAgent):
    """
    NeuralNetQAgent
    You should only have to overwrite getQValue
    and update. All other QLearningAgent functions
    should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        """
        Initialise our neural network here. Instead of figuring out
        features, we will throw in our entire state as a list
        We don't know the size of our game here, so we can't create our
        neural network yet. We'll delay the creation of our neural network
        until the first time we are asked about it
        """
        self.nnet = None

    def getQValue(self, state, action):
        """
        Should return Q(state,action) by asking our neural network
        """
        if self.nnet is None:
            self.nnet = NeuralNetwork(state)
        return self.nnet.predict(state, action)

    def update(self, state, action, nextState, reward):
        """
        Should update our neural network based on transition
        """
        if self.nnet is None:
            self.nnet = NeuralNetwork(state)

        "*** YOUR CODE HERE ***"

        qval = self.nnet.predict(state, action)

        # Get max_Q(S',a)
        newQ = self.nnet.predict(nextState, action)
        maxQ = np.max(newQ)
        y = np.zeros((1, 4))
        y[:] = qval[:]

        if not state.isTerminal():  # non-terminal state
            update = (reward + (self.discount * maxQ))
        else:  # terminal state
            update = reward
        y[0][action] = update  # target output

        self.nnet.update(state, action, y)


class NeuralNetwork:
    def __init__(self, state):
        """
        We can only initialise our neural network with a state
        so we can determine the size of our world
        Size will be:
        a) walls + food + ghost: 3 * width * height
        b) pacmanPosition: width * height
        c) nextPacmanPosition: width * height
        """
        walls = state.getWalls()
        self.width = walls.width
        self.height = walls.height
        self.size = 5 * self.width * self.height

        "*** YOUR CODE HERE ***"
        self.model = Sequential()
        self.model.add(Dense(164, init='lecun_uniform', input_shape=(self.size,)))
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        self.model.add(Dense(150, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        self.model.add(Dense(4, init='lecun_uniform'))
        self.model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    def predict(self, state, action):
        reshaped_state = self.reshape(state, action)
        "*** YOUR CODE HERE ***"
        return reshaped_state  # What in the world are you even supposed to do with this?

    def update(self, state, action, y):
        reshaped_state = self.reshape(state, action)

        "*** YOUR CODE HERE ***"
        self.model.fit(reshaped_state, y, batch_size=1, nb_epoch=1, verbose=1)

    def reshape(self, state, action):
        """
        a) walls + food + ghost: 3 * width * height
        b) pacmanPosition: width * height
        c) nextPacmanPosition: width * height
        """

        reshaped_state = np.empty((1, 2 * self.size))
        food = state.getFood()
        walls = state.getWalls()
        for x in range(self.width):
            for y in range(self.height):
                reshaped_state[0][x * self.width + y] = int(food[x][y])
        reshaped_state[0][self.size + x * self.width + y] = int(walls[x][y])
        ghosts = state.getGhostPositions()
        ghost_states = np.zeros((1, self.size))
        for g in ghosts:
            ghost_states[0][int(g[0] * self.width + g[1])] = int(1)
        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pacman_state = np.zeros((1, self.size))
        pacman_state[0][int(x * self.width + y)] = 1
        pacman_nextState = np.zeros((1, self.size))
        pacman_nextState[0][int(next_x * self.width + next_y)] = 1
        reshaped_state = np.concatenate((reshaped_state, ghost_states, pacman_state,
                                         pacman_nextState), axis=1)
        return reshaped_state
