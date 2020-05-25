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

def mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst)/len(lst)

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if newPos in successorGameState.getGhostPositions() or action == 'Stop':
            return -float('inf')
        if newPos in currentGameState.getFood().asList():
            return +float('inf')
        numFood = len(newFood.asList())
        scaredTime = min(newScaredTimes)
        closestDistFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])
        return scaredTime - numFood - closestDistFood

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        scores = []
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            scores.append(self.minimax(successorState, self.depth, 1))
        return gameState.getLegalActions(self.index)[scores.index(max(scores))]

    def minimax(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxScore = -float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimax(successorState, depth, 1)
                maxScore = max(newScore, maxScore)
            return maxScore
        elif index+1 < gameState.getNumAgents():
            minScore = +float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimax(successorState, depth, index+1)
                minScore = min(newScore, minScore)
            return minScore
        else:
            minScore = +float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimax(successorState, depth-1, 0)
                minScore = min(newScore, minScore)
            return minScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxScore = -float("inf")
        alpha = -float("inf")
        beta = +float("inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            newScore = self.minimaxAlphaBeta(successorState, self.depth, 1, alpha, beta)
            if newScore > maxScore:
                maxScore = newScore
                maxAction = action
            alpha = max(alpha, maxScore)
        return maxAction

    def minimaxAlphaBeta(self, gameState, depth, index, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxScore = -float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimaxAlphaBeta(successorState, depth, 1, alpha, beta)
                maxScore = max(newScore, maxScore)
                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)
            return maxScore
        elif index+1 < gameState.getNumAgents():
            minScore = +float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimaxAlphaBeta(successorState, depth, index+1, alpha, beta)
                minScore = min(newScore, minScore)
                if minScore < alpha:
                    return minScore
                beta = min(beta, minScore)
            return minScore
        else:
            minScore = +float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.minimaxAlphaBeta(successorState, depth-1, 0, alpha, beta)
                minScore = min(newScore, minScore)
                if minScore < alpha:
                    return minScore
                beta = min(beta, minScore)
            return minScore

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
        scores = []
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            scores.append(self.expectimax(successorState, self.depth, 1))
        return gameState.getLegalActions(self.index)[scores.index(max(scores))]

    def expectimax(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxScore = -float('inf')
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.expectimax(successorState, depth, 1)
                maxScore = max(newScore, maxScore)
            return maxScore
        elif index+1 < gameState.getNumAgents():
            scores = []
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.expectimax(successorState, depth, index+1)
                scores.append(newScore)
            return mean(scores)
        else:
            scores = []
            for action in gameState.getLegalActions(index):
                successorState = gameState.generateSuccessor(index, action)
                newScore = self.expectimax(successorState, depth-1, 0)
                scores.append(newScore)
            return mean(scores)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <base on currentGameState.getScore(), get closer to foods, stay away from ghosts and eat as many dot and capsules as possible, use some fine tuned parameter to combine them>
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    dist2Foods = [+float('inf')]
    for foodPos in currentGameState.getFood().asList():
        dist2Foods.append(util.manhattanDistance(currentPos, foodPos)) 

    dist2Ghosts = [1]
    for ghostPos in currentGameState.getGhostPositions():
        dist2Ghosts.append(util.manhattanDistance(currentPos, ghostPos))

    return 0.5*currentGameState.getScore() + 1.5/min(dist2Foods) - 1.0/max(dist2Ghosts) + 2.0/(currentGameState.getNumFood()+1) + 10.0/(len(currentGameState.getCapsules())+1)

# Abbreviation
better = betterEvaluationFunction
