# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def search(problem, structure):
    """
    We realise that the different search algorithms are in fact really similar in nature.
    The main algorithm is the same and the only difference is the type of data structure
    used to do the search algorithm.

    As such, to make this simpler, we use this search function which takes in the problem
    and the type of data structure as parameters, so that all we have to do for the different
    search algorithms is pass in a different data structure.
    """

    searched = []  # To contain the nodes that have already been searched

    # We store the different paths in lists, with each node being a list on its own
    structure.push([[problem.getStartState(), "STOP", 1]])

    while not structure.isEmpty():  # In the case where we do not find the goal state
        path = structure.pop()  # A different path will be returned based on the data structure
        last = path[-1][0]  # This is the last node of the path which we expand from

        if problem.isGoalState(last):
            #  Return the directions if we have found the goal state
            return [direction[1] for direction in path[1:]]

        if last not in searched:  # Ensure that we do not search through the same node more than once
            searched.append(last)  # Put the node in the searched lists so we know we have already expanded it

            for i in problem.getSuccessors(last):
                # Create a new path for each of the successors the current node has before putting it into the structrue
                new_path = path[:]
                new_path.append(i)
                structure.push(new_path)

    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # We use a stack here so that we keep expanding the same path
    structure = util.Stack()
    return search(problem, structure)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # A queue allows us to search outwards one node at a time
    structure = util.Queue()
    return search(problem, structure)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Compute the cost of the path by feeding the directions of the path into the problem.getCostOfActions function
    Function = lambda path: problem.getCostOfActions([node[1] for node in path[1:]])

    # We use a priority queue so that we can prioritise paths with a lower cost
    structure = util.PriorityQueueWithFunction(Function)
    return search(problem, structure)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # This function is the same as UCS except we add the heuristic cost to it,
    # passing in the last node and problem into the function we get in the parameters
    Function = lambda path: problem.getCostOfActions([node[1] for node in path[1:]]) \
                            + heuristic(path[-1][0], problem)

    # We use a priority queue so that we can prioritise paths with a lower forwards and backwards cost
    structure = util.PriorityQueueWithFunction(Function)
    return search(problem, structure)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
