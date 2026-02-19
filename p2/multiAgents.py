"""multiAgents.py - Multi-Agent Search Algorithms for Pacman
===========================================================

This module implements various multi-agent search algorithms for the Pacman game,
including reflex agents, minimax, alpha-beta pruning, and expectimax search.

The module provides agent classes that:
- Make decisions based on state evaluation functions
- Implement adversarial search algorithms
- Model both deterministic and probabilistic opponent behavior
- Search to configurable depths using evaluation heuristics

Key Classes:
    ReflexAgent: Makes decisions using state evaluation heuristics
    MinimaxAgent: Implements minimax search algorithm
    AlphaBetaAgent: Implements alpha-beta pruning search
    ExpectimaxAgent: Implements expectimax probabilistic search

Usage:
    This module is used by the Pacman game to create AI agents. Agents can be
    selected and configured via command line arguments.

Author: George Rudolph
Date: 14 Nov 2024
Major Changes:
1. Added type hints throughout the codebase for better code clarity and IDE support
2. Improved docstrings with detailed descriptions and Args/Returns sections
3. Enhanced code organization with better function and variable naming

This code runs on Python 3.13

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import random, math
import util
from util import manhattanDistance
from game import Agent, Directions
from typing import List, Tuple, Any
from pacman import GameState

class ReflexAgent(Agent):
    """A reflex agent that chooses actions by examining alternatives via a state evaluation function.
    
    This agent evaluates each possible action using a heuristic evaluation function and selects
    among the best options. The evaluation considers factors like:
    - Distance to ghosts (avoiding them)
    - Score improvements
    - Distance to food
    - Maintaining movement direction
    """

    def getAction(self, gameState: GameState) -> str:
        """Choose among the best actions according to the evaluation function.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: A direction from Directions.{North, South, West, East, Stop}
            
        The method collects legal moves, scores them using the evaluation function,
        and randomly selects among those with the best score.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """Evaluate the desirability of a game state after taking an action.
        
        Args:
            currentGameState: The current game state
            action: The proposed action
            
        Returns:
            float: A score where higher numbers are better, using values 8,4,2,1,0
            that are bitwise orthogonal (powers of 2)
            
        The function evaluates states based on:
        - Avoiding ghosts (returns 0 if too close)
        - Score improvements (returns 8)
        - Getting closer to food (returns 4) 
        - Maintaining direction (returns 2)
        - Default case (returns 1)
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 1
        danger = 1
        # Checks to see if the ghost is closer then danger and if there scared times is 0
        for ghostState, scaredTimes in zip(newGhostStates, newScaredTimes):            
            dist = manhattanDistance(newPos, ghostState.getPosition())
            if scaredTimes == 0 and dist <= danger:
                return 0
        
        # Checks all the food and rewards the closest food
        foodList = newFood.asList()
        if foodList:
            closestFood = 100
            for food in foodList:
                closestFoodDistance = manhattanDistance(newPos, food)
                if closestFoodDistance < closestFood:
                    closestFood = closestFoodDistance
            if closestFood > 0:
                score += 4 / closestFood

        if successorGameState.getScore() > currentGameState.getScore():
            score += 8
        
        return score




def scoreEvaluationFunction(currentGameState: GameState) -> float:
    """Return the score of the state for use with adversarial search agents.
    
    Args:
        currentGameState: The game state to evaluate
        
    Returns:
        float: The score displayed in the Pacman GUI
        
    This is the default evaluation function for adversarial search agents.
    Not intended for use with reflex agents.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """Base class for adversarial search agents (minimax, alpha-beta, expectimax).
    
    This abstract class provides common functionality for multi-agent searchers.
    It should not be instantiated directly, but rather extended by concrete
    agent implementations.
    
    Attributes:
        index: Agent index (0 for Pacman)
        evaluationFunction: Function used to evaluate game states
        depth: Maximum depth of search tree
    """

    def __init__(self, evalFn: str = 'scoreEvaluationFunction', depth: str = '2') -> None:
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax agent that implements adversarial search.
    
    This agent uses minimax search to determine the optimal action by considering
    the worst case scenario at each level.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The optimal action according to minimax search
            
        Uses self.depth and self.evaluationFunction to determine the best action
        by considering the worst-case scenario at each level.
        """

        "*** YOUR CODE HERE ***"

        ghostIndices = list(range(1, gameState.getNumAgents()))
        firstGhost = ghostIndices[0]
        lastGhost = ghostIndices[-1]

        def isTerminal(state,depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        def minValue(state,depth,ghostIndex):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            value = float('inf')
            for action in state.getLegalActions(ghostIndex):
                nextState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == lastGhost:
                    value = min(value, maxValue(nextState, depth +1))
                else:
                    value = min(value, minValue(nextState,depth, ghostIndex + 1))
            return value
            
        def maxValue(state,depth):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            value = float('-inf')
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor( 0,action)
                value = max(value, minValue(nextState, depth, firstGhost))
            return value
        
        bestAction = None
        bestScore = float("-inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            score = minValue(nextState, 0, firstGhost)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Minimax agent with alpha-beta pruning optimization.
    
    This agent implements minimax search with alpha-beta pruning to more efficiently
    explore the game tree by pruning branches that cannot affect the final decision.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action using alpha-beta pruning.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The optimal action according to alpha-beta pruning
            
        Pacman is always the max agent, ghosts are always min agents.
        At depth 0, max_value returns an action. At other depths, it returns a value.
        """

        "*** YOUR CODE HERE ***"
        ghostIndices = list(range(1, gameState.getNumAgents()))
        firstGhost = ghostIndices[0]
        lastGhost = ghostIndices[-1]

        def isTerminal(state,depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        def minValue(state,depth,ghostIndex,alpha,beta):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            value = float('inf')

            for action in state.getLegalActions(ghostIndex):
                nextState = state.generateSuccessor(ghostIndex, action)

                if ghostIndex == lastGhost:
                    value = min(value, maxValue(nextState, depth + 1, alpha, beta))
                else:
                    value = min(value, minValue(nextState,depth, ghostIndex + 1,alpha,beta))

                if value < alpha:
                    return value
                beta = min(beta,value)
            return value
            
        def maxValue(state,depth,alpha,beta):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            value = float('-inf')

            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor( 0,action)
                value = max(value, minValue(nextState, depth, firstGhost,alpha,beta))

                if value > beta:
                    return value
                alpha = max(alpha,value)
            return value
        
        bestAction = None
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            score = minValue(nextState, 0, firstGhost,alpha,beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """An agent that uses expectimax search to make decisions.
    
    This agent models ghosts as choosing uniformly at random from their legal moves.
    It uses expectimax search to find optimal actions against probabilistic opponents.
    
    The agent searches to a fixed depth using a supplied evaluation function.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the expectimax action using self.depth and self.evaluationFunction.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The selected action (one of Directions.{North,South,East,West,Stop})
            
        All ghosts are modeled as choosing uniformly at random from their legal moves.
        """
        
        "*** YOUR CODE HERE ***"

        ghostIndices = list(range(1, gameState.getNumAgents()))
        firstGhost = ghostIndices[0]
        lastGhost = ghostIndices[-1]

        def isTerminal(state,depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        def expValue(state,depth,ghostIndex):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            sumValue = 0
            numActions = 0
            
            for action in state.getLegalActions(ghostIndex):
                nextState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == lastGhost:
                    value = maxValue(nextState, depth + 1)
                else:
                    value = expValue(nextState,depth, ghostIndex + 1)
                sumValue += value
                numActions += 1
            return sumValue / numActions
            
        def maxValue(state,depth):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            value = float('-inf')

            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor( 0,action)
                temp = expValue(nextState,depth,firstGhost)
                value = max(value,temp)
            return value
        
        bestAction = None
        bestScore = float("-inf")

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            score = expValue(nextState, 0, firstGhost)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(game_state: GameState) -> float:
    """A more sophisticated evaluation function for Pacman game states.
    
    This function evaluates states by combining the game score with a penalty
    based on distance to the closest food pellet. The penalty uses the reciprocal
    of the distance to give higher penalties to food that is farther away.
    
    Args:
        game_state: The game state to evaluate
        
    Returns:
        float: The evaluation score where higher values are better
    """
    
    "*** YOUR CODE HERE ***"
    newPos = game_state.getPacmanPosition()
    newFood = game_state.getFood()
    newGhostStates = game_state.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    INF = float("inf")
    WEIGHT_FOOD = 10.0
    WEIGHT_GHOST = -10.0
    WEIGHTED_SCARED_GHOST = 100.0

    score = game_state.getScore()

    foodList = newFood.asList()
    distanceList = []
    if foodList:
        for food in foodList:
            distanceList.append(util.manhattanDistance(newPos,food))
    
    if len(distanceList) > 0:
        score += WEIGHT_FOOD / min(distanceList)
    else:
        score += WEIGHT_FOOD

    for ghostState, scaredTimes in zip(newGhostStates, newScaredTimes):            
            dist = manhattanDistance(newPos, ghostState.getPosition())
            if dist > 0:
                if scaredTimes > 0:
                    score += WEIGHTED_SCARED_GHOST / dist
                else:
                    score += WEIGHT_GHOST / dist
            else:
                return float("-inf")
    return score
    
# Abbreviation
better = betterEvaluationFunction

