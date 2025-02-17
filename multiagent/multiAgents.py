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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood().asList()
        ghostPositions = successorGameState.getGhostPositions()

        # Food distances
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood]
        closestFoodDistance = min(foodDistances) if foodDistances else 1  # distance to the closest food if food exists


        # Ghost distances
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        closestGhostDistance = min(ghostDistances) if ghostDistances else float('inf')


        ghostThreshold = 2  # Threshold for being "too close" to a ghost
        ghostPenalty = 0
        if closestGhostDistance < ghostThreshold:
            ghostPenalty = -10 / (closestGhostDistance + 1)  # Strong penalty for danger, don t go there , too risky

        # Reward for food
        foodReward = 10 / closestFoodDistance  # Prioritize getting closer to food

        #Obs: 10 / closestFoodDistance or -10 / (closestGhostDistance + 1) are just some weights that can be changed

        # Overall evaluation
        evaluation = successorGameState.getScore() + foodReward + ghostPenalty
        return evaluation


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        maxScore = float('-inf')
        actions=gameState.getLegalActions(0) #Pacman muta primul
        for a in actions:
            successor=gameState.generateSuccessor(0, a)
            # Start with depth = 0 and the agent index = 1 (first ghost)
            currentResult = self.minValue(successor, 0, 1)
            if currentResult > maxScore:
                maxScore = currentResult
                maxAction = a
        return maxAction


        util.raiseNotDefined()

    def minValue(self, gameState, currDepth, ghost):
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(ghost)
        successors = []
        for a in actions:
            successors.append(gameState.generateSuccessor(ghost, a))
        ghosts=gameState.getNumAgents()
        mini=float('inf')
        for s in successors:
            if ghost < ghosts-1:
                # mai sunt fantome care trebuie sa joace
                mini=min(mini,self.minValue(s, currDepth, ghost+1))
            else:
                # crestem depth cand e randul lui Pacman din nou
                mini=min(mini,self.maxValue(s,currDepth+1))
        return mini


    def maxValue(self, gameState, currDepth ):
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        successors = []
        for a in actions:
            successors.append(gameState.generateSuccessor(0, a))
        maxi=float('-inf')
        for s in successors:
            maxi=max(maxi,self.minValue(s, currDepth,1))
        return maxi


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """



    def getAction(self, gameState: GameState):
            """
            Returns the minimax action using self.depth and self.evaluationFunction
            """
            maxScore = float('-inf')
            alpha = float('-inf')
            beta = float('inf')


            actions = gameState.getLegalActions(0)  # Pacman moves first
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                # Start with depth = 0 and the agent index = 1 (first ghost)
                currentScore = self.minValue(successor, 1, 0, alpha, beta)
                if currentScore > maxScore:
                    maxScore = currentScore
                    bestAction = action
                alpha = max(alpha, maxScore)

            return bestAction

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
            # Terminal state check
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            minScore = float('inf')
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:  # Last ghost
                    minScore = min(minScore, self.maxValue(successor, depth + 1, alpha, beta))
                else:  # Next ghost
                    minScore = min(minScore, self.minValue(successor, agentIndex + 1, depth, alpha, beta))

                if minScore < alpha:
                    return minScore
                beta = min(beta, minScore)

            return minScore

    def maxValue(self, gameState, depth, alpha, beta):
            # Terminal state check
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)  # Pacman's actions

            maxScore = float('-inf')
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                maxScore = max(maxScore, self.minValue(successor, 1, depth, alpha, beta))

                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)

            return maxScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
