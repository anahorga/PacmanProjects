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
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class CaleParinte:
    def __init__(self,state,parent=None,action=None,cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
def BuildPath(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1]#inversam calea pt a merge de la start la scop
def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    """
    "*** YOUR CODE HERE ***"

    """  prima varianta
  frontier=util.Stack()
    expanded=[]
    frontier.push((problem.getStartState(),[]))
    while not frontier.isEmpty():
         currentstate, path=frontier.pop()
         if problem.isGoalState(currentstate):
             #print("am gasit solutia")
             return path
         else:
             expanded.append(currentstate)
             for nextstate , action , _ in problem.getSuccessors(currentstate):
                 #print(f"push in frontier{nextstate}")
                 if nextstate not in expanded:
                     frontier.push((nextstate,path+[action]))

    """
    frontier = util.Stack()
    expanded = []
    start_node=CaleParinte(problem.getStartState())
    frontier.push(start_node)
    while not frontier.isEmpty():
        current_node= frontier.pop()
        if problem.isGoalState(current_node.state):
            return BuildPath(current_node)

        else:
            expanded.append(current_node.state)
            for nextstate, action, _ in problem.getSuccessors(current_node.state):
                #print(f"push in frontier{nextstate}")
                if nextstate not in expanded:
                    next_node=CaleParinte(nextstate,current_node,action)
                    frontier.push(next_node)
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    """ varinta mai neeficienta
    frontier = util.Queue()
    expanded = []
    frontier.push((problem.getStartState(), []))
    while not frontier.isEmpty():
        currentstate, path = frontier.pop()
        if problem.isGoalState(currentstate):
            # print("am gasit solutia")
            return path
        else:
            expanded.append(currentstate)
            for nextstate, action, _ in problem.getSuccessors(currentstate):
                # print(f"push in frontier{nextstate}")
                if nextstate not in expanded :
                    frontier.push((nextstate, path + [action]))"""

    frontier = util.Queue()
    expanded = []
    in_frontiera=set()
    start_node = CaleParinte(problem.getStartState())
    frontier.push(start_node)
    in_frontiera.add(problem.getStartState())
    while not frontier.isEmpty():
        current_node = frontier.pop()
        in_frontiera.remove(current_node.state)
        if problem.isGoalState(current_node.state):
            # print("am gasit solutia")
            return BuildPath(current_node) # inversam calea pt a merge de la start la scop

        else:
            expanded.append(current_node.state)
            for nextstate, action, _ in problem.getSuccessors(current_node.state):
                # print(f"push in frontier{nextstate}")
                if nextstate not in expanded and nextstate not in in_frontiera:
                    next_node = CaleParinte(nextstate, current_node, action)
                    frontier.push(next_node)
                    in_frontiera.add(nextstate)
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier=util.PriorityQueue()
    expanded = dict()
    start_node = CaleParinte(problem.getStartState())
    frontier.push(start_node,0)
    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_cost = current_node.cost
        if problem.isGoalState(current_node.state):
            return BuildPath(current_node)
        else:
             expanded[current_node.state] = current_cost
             for nextstate, action, cost in problem.getSuccessors(current_node.state):
                total_cost = current_cost + cost
                if nextstate not in expanded or total_cost <expanded[nextstate]:
                    expanded[nextstate] = total_cost
                    frontier.push(CaleParinte(nextstate,current_node,action,total_cost),total_cost)

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    frontier = util.PriorityQueue()
    expanded = dict()
    start_node = CaleParinte(problem.getStartState())
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_cost = current_node.cost
        if problem.isGoalState(current_node.state):
            return BuildPath(current_node)
        else:
            expanded[current_node.state] = current_cost
            for nextstate, action, cost in problem.getSuccessors(current_node.state):
                total_cost = current_cost + cost
                if nextstate not in expanded or total_cost < expanded[nextstate]:
                    expanded[nextstate] = total_cost
                    frontier.push(CaleParinte(nextstate, current_node, action, total_cost), total_cost+heuristic(nextstate, problem))

    return []

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
