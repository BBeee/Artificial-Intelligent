# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    #basically returns all the actions possible from that state
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    #calculate the score of each state according to some evaluation function
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    #collect the inidices of the best scored state
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    print(' Score: '+str(scores[chosenIndex]))
    print('\n returned move: '+str(legalMoves[chosenIndex]))

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currGameState, pacManAction):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    # generate the next state for that action from that state
    #print('\n Inside evaluation function')
    nextGameState = currGameState.generatePacmanSuccessor(pacManAction)
    #print('next game state: '+str(nextGameState))
    newPos = nextGameState.getPacmanPosition()
    #print('new position: '+str(newPos))
    oldFood = currGameState.getFood()
    #print(' old food: '+str(oldFood))
    newGhostStates = nextGameState.getGhostStates()
    #print(' new ghost states: '+ str(newGhostStates))
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #print(' new scared times: '+str(newScaredTimes))
    
    "*** YOUR CODE HERE ***"
    #print('returned value: '+str(nextGameState.getScore()))
    
    
    #logic:
    #reward states farther from ghosts
    #penalize states with no food
    #reward states that move toward capsules
    
    
    if nextGameState.isWin():
        return float("inf")
    
    newFood = nextGameState.getFood()
    oldFood_len = oldFood.count()
    
    activeGhosts = []
    scaredGhosts = []
    
    for ghost in newGhostStates:
        if not ghost.scaredTimer:
            activeGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)
    
    #score = nextGameState.getScore()
    closest_food = []
    if len(newFood.asList()) == oldFood_len:
        dis = 10000
        for food in newFood.asList():
            if util.manhattanDistance(food, newPos) < dis:
                dis = util.manhattanDistance(food, newPos)
                closest_food = food
        closest_dist = dis
    else:
        dis = 0
        
    dist_closestscaredghost = 0
    for i, ghost in enumerate(newGhostStates):
        if ghost in activeGhosts:
            dis += 4** (2- util.manhattanDistance(ghost.getPosition(), newPos))
            #dis -= util.manhattanDistance(ghost.getPosition(), newPos)
        elif ghost in scaredGhosts:
            dist = util.manhattanDistance(ghost.getPosition(), newPos)
            if newScaredTimes[i] != 0:
                if len(scaredGhosts)==1:
                    dis -= 4**(2-dist)
                    #dis-= 150
                if i == 0:
                    dist_closestscaredghost = dist
                if i==1 and dist < dist_closestscaredghost:
                    dis -= 4**(2-dist)
                    #dis-= 150
                if i==1 and dist> dist_closestscaredghost:
                    dis -= 4**(2-dist_closestscaredghost)
                    #dis -= 150

    if pacManAction == Directions.STOP:
        dis += 3
        if util.manhattanDistance(closest_food, newPos) < closest_dist:
            dis -= 4**(2-util.manhattanDistance(closest_food, newPos)) - 5

    capsules = currGameState.getCapsules()
    if newPos in capsules:
        dis -= 10


    return -dis








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

  def getAction(self, currGameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      currGameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      currGameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      currGameState.getNumAgents():
        Returns the total number of agents in the game
    """
            
    #Pacman
    def maxvalue(gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        possibleActions = gameState.getLegalActions(0)
        numGhosts = gameState.getNumAgents() - 1
        value = -1000
        for action in possibleActions:
            value = max(value, minvalue(gameState.generateSuccessor(0,action), numGhosts, 1, depth-1))
        return value

    #Ghost
    def minvalue(gameState,  numGhosts, whichGhost, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        possibleActions = gameState.getLegalActions(whichGhost)
        value = 1000
        if whichGhost == numGhosts:
            for action in possibleActions:
                value = min(value, maxvalue(gameState, depth-1))
        else:
            for action in possibleActions:
                value = min(value, minvalue(gameState, numGhosts, whichGhost+1, depth))
        return value



    possibleActions = currGameState.getLegalActions(0)
    numGhosts = currGameState.getNumAgents() - 1
    #Initializing score with negative infinite number   -(float("inf"))
    score = -1000
    nextAction = Directions.STOP
    for action in possibleActions:
        nextState = currGameState.generateSuccessor(0, action)
        if nextState.isWin():
            return action
        value = minvalue(nextState, numGhosts, 1, self.depth)
        if value > score:
            score = value
            nextAction = action
    return nextAction





class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, currGameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #Pacman
    def maxvalue(gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        possibleActions = gameState.getLegalActions(0)
        numGhosts = gameState.getNumAgents() - 1
        value = -1000
        for action in possibleActions:
            value = max(value, minvalue(gameState.generateSuccessor(0,action), numGhosts, 1, depth-1, alpha, beta))
            alpha = max(value, alpha)
            if value >= beta:
                return value
        return value


    #Ghost
    def minvalue(gameState,  numGhosts, whichGhost, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        possibleActions = gameState.getLegalActions(whichGhost)
        value = 1000
        for action in possibleActions:
            nextState = gameState.generateSuccessor(whichGhost, action)
            if whichGhost == numGhosts:
                value = min(value, maxvalue(nextState, depth-1, alpha, beta))
                beta = min(value, beta)
                if value <= alpha:
                    return value
            else:
                value = min(value, minvalue(nextState, numGhosts, whichGhost+1, depth, alpha, beta))
                beta = min(value, beta)
                if value <= alpha:
                    return value
        return value


    possibleActions = currGameState.getLegalActions(0)
    numGhosts = currGameState.getNumAgents() - 1
    #Initializing score with negative infinite number   -(float("inf"))
    score = -1000
    alpha =  -(float("inf"))
    beta = float("inf")
    for action in possibleActions:
        nextState = currGameState.generateSuccessor(0, action)
        if nextState.isWin():
            return action
        value = minvalue(nextState, numGhosts, 1, self.depth, alpha, beta)
        if value > score:
            score = value
            nextAction = action
        if value >= beta:
            return nextAction
        alpha = max(alpha, score)
    return nextAction



class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, currGameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currGameState):

    pos = currGameState.getPacmanPosition()
    currentScore = scoreEvaluationFunction(currGameState)
        
    if currGameState.isWin():return float("inf")
    if currGameState.isLose():return -float("inf")
                
    foodlist = currGameState.getFood().asList()
    dis = 10000
    for food in foodlist:
        if util.manhattanDistance(food, newPos) < dis:
            dis = util.manhattanDistance(food, pos)
                                    
    capsules_left = currGameState.getCapsules()
    food_left = len(foodlist)
                                            
    scaredGhosts, activeGhosts = [], []
                                                
    for ghost in currGameState.getGhostStates():
        if not ghost.scaredTimer():
            activeGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)
                                                                    
    def getDistance(ghost):
        return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghost)
                                                                            
    closest_activeghost = 0
    closest_scaredghost = 0

    if activeGhosts:
        closest_activeghost = min(getDistance(activeGhosts))
    else:
        closest_activeghost = float("inf")
    closest_activeghost = max(closest_activeghost, 5)

    if scaredGhosts:
        closest_scaredghost = min(getDistance(scaredGhosts))
    else:
        closest_scaredghost = 0
    score = 1 * currentScore -1.5 * dis -2 * (1./closest_activeghost) -2 * closest_scaredghost-20 * capsules_left -4 * food_left
    return score


# Abbreviation
better = betterEvaluationFunction



