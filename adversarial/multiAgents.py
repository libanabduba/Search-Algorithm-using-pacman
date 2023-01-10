# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"

        score = 0  # This will be the output of the function

        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:  # Check if pacman is too close to the ghost
                if ghost.scaredTimer:  # If the ghost can be eaten, then pacman should eat the ghost
                    score += 20
                else:
                    return -1  # The pacman should avoid the ghost at all costs!

        food = currentGameState.getFood()

        # Search through all the food in the gameState
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    if newPos == (x, y):  # If the food is next to the pacman, pacman should eat it
                        score += 20
                    else:
                        # We use the reciprocal of manhattanDistance of the food to the pacman so that the further away
                        # the food, the less the score.  This allows pacman to go to food that is nearer to it first
                        score += 10/manhattanDistance(newPos, (x, y))

        # Capsules are important as they could potentially earn more points
        for capsule in currentGameState.getCapsules():
            if newPos == capsule:  # Eat the capsule if pacman is just next to it
                score += 20
            else:
                # Similarly to the food, the reciprocal of the manhattanDistance between the capsule and pacman
                # ensures that we do not go for the capsule if it is too far away
                score += 10/manhattanDistance(capsule, newPos)

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)  # Collect legal moves
        agents = range(gameState.getNumAgents())  # Store the agents in a list

        # These are the final minimax scores for each of the possible legal actions
        # The implementation of choosing the best action is pretty much the same as the one for the reflex agent
        scores = [self.value(gameState.generateSuccessor(0, action), agents[1:], 0) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def value(self, gameState, agents, depth):
        """
        This is the function that determines whether the next agent is a MIN or a MAX, or if the gameState is terminal,
        or has reached maximum depth.  It then calls on the relevant functions.

        We store all the agents in a list, such that everytime we go through one cycle of that list, we have go down
        one more depth.
        """
        if len(agents) == 0:  # if there are no more agents in that cycle, reinitialise the agent variable
            agents = range(gameState.getNumAgents())
            depth += 1

        if gameState.isWin() or gameState.isLose() or self.depth == depth:  # Check whether the gameState is terminal or maximum depth reached
            return self.evaluationFunction(gameState)
        elif agents[0] == 0:  # This is the pacman agent, so we use a max functoin
            return self.max_value(gameState, agents, depth)
        else:
            return self.min_value(gameState, agents, depth)

    def max_value(self, gameState, agents, depth):
        """
        We first find all the possible actions the pacman can make, before calling the value function again for each of
        the successor states based on the actions possible.  Then we find the maximum value since this is a MAX function
        """
        v = -float("inf")

        legalMoves = gameState.getLegalActions(agents[0])
        successors = [gameState.generateSuccessor(agents[0], action) for action in legalMoves]
        for successor in successors:
            v = max(v, self.value(successor, agents[1:], depth))

        return v

    def min_value(self, gameState, agents, depth):
        """
        Same implementaion as max_value function except float starts off as positive infinity, and we find the min v
        """
        v = float("inf")

        legalMoves = gameState.getLegalActions(agents[0])
        successors = [gameState.generateSuccessor(agents[0], action) for action in legalMoves]
        for successor in successors:
            v = min(v, self.value(successor, agents[1:], depth))

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        agents = range(gameState.getNumAgents())

        """
        Unlike the minimax algorithm before, we have to consider 2 more variables, alpha and beta.  As such, we have to
        change our implmentation slightly, so that we can edit the alpha value after each call of self.value.
        
        Thus, instead of using list comprehension to get all the scores at once as above, we loop through all the
        possible legalMoves one by one.
        """

        alpha = -float("inf")  # Best score for the pacman agent
        beta = float("inf")  # Best score for the ghosts
        best_action = "Stop"  # Best action based on alpha, to be returned by this function

        for action in legalMoves:  # Loop through all the possible legalMoves intead of list comprehension
            successor = gameState.generateSuccessor(0, action)

            v = self.value(successor, agents[1:], 0, alpha, beta)

            if v > alpha:  # Check to see if we have found a better score
                alpha = v
                best_action = action

        return best_action

    def value(self, gameState, agents, depth, alpha, beta):
        """
        Basically the same as before, just that we include alpha and beta as parameters
        """
        if len(agents) == 0:
            agents = range(gameState.getNumAgents())
            depth += 1

        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return self.evaluationFunction(gameState)
        elif agents[0] == 0:
            return self.max_value(gameState, agents, depth, alpha, beta)
        else:
            return self.min_value(gameState, agents, depth, alpha, beta)

    def max_value(self, gameState, agents, depth, alpha, beta):
        """
        Both the max_value and min_value functions are similar to before, except we use a loop instead of list
        comprehension so that we can check v against alpha/beta each time we loop through another gameState, so that we
        do not need to unecessarily search through some gameStates
        """
        v = -float("inf")

        legalMoves = gameState.getLegalActions(agents[0])
        for action in legalMoves:
            successor = gameState.generateSuccessor(agents[0], action)

            v = max(v, self.value(successor, agents[1:], depth, alpha, beta))

            # Pruning part
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, gameState, agents, depth, alpha, beta):
        """
        Similar to max_value function but the opposite
        """
        v = float("inf")

        legalMoves = gameState.getLegalActions(agents[0])
        for action in legalMoves:
            successor = gameState.generateSuccessor(agents[0], action)

            v = min(v, self.value(successor, agents[1:], depth, alpha, beta))

            # Pruning part
            if v < alpha:
                return v
            beta = min(beta, v)

        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)  # Collect legal moves
        agents = range(gameState.getNumAgents())  # Store the agents in a list

        # These are the final minimax scores for each of the possible legal actions
        # The implementation of choosing the best action is pretty much the same as the one for the reflex agent
        scores = [self.value(gameState.generateSuccessor(0, action), agents[1:], 0) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        try:
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        except IndexError:
            return Directions.STOP
        return legalMoves[chosenIndex]

    def value(self, gameState, agents, depth):
        """
        This is the function that determines whether the next agent is a MIN or a MAX, or if the gameState is terminal,
        or has reached maximum depth.  It then calls on the relevant functions.

        We store all the agents in a list, such that everytime we go through one cycle of that list, we have go down
        one more depth.
        """
        if len(agents) == 0:  # if there are no more agents in that cycle, reinitialise the agent variable
            agents = range(gameState.getNumAgents())
            depth += 1

        if gameState.isWin() or gameState.isLose() or self.depth == depth:  # Check whether the gameState is terminal or maximum depth reached
            return self.evaluationFunction(gameState)
        elif agents[0] == 0:  # This is the pacman agent, so we use a max functoin
            return self.max_value(gameState, agents, depth)
        else:
            return self.exp_value(gameState, agents, depth)

    def max_value(self, gameState, agents, depth):
        """
        We first find all the possible actions the pacman can take, before calling the value function again for each of
        the successor states based on the actions possible.  Then we find the maximum value since this is a MAX function
        """
        v = -float("inf")

        legalMoves = gameState.getLegalActions(agents[0])
        successors = [gameState.generateSuccessor(agents[0], action) for action in legalMoves]
        for successor in successors:
            v = max(v, self.value(successor, agents[1:], depth))

        return v

    def exp_value(self, gameState, agents, depth):
        """
        We sum up the value of all the possible successors of the gameState, before dividing by the number of successors
        to find out the expected score.  This is based on an equal probability of each action being chosen by the ghost.
        """
        legalMoves = gameState.getLegalActions(agents[0])
        successors = [gameState.generateSuccessor(agents[0], action) for action in legalMoves]
        total_score = float(0)

        for successor in successors:
            v = float(self.value(successor, agents[1:], depth))
            total_score += v

        return total_score/float(len(legalMoves))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction
