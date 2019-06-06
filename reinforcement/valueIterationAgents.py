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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        for i in range(0, self.iterations):
            newValues = util.Counter()
            for state in states:
                optimalAction = self.computeActionFromValues(state)
                v_kplus1 = self.computeQValueFromValues(state, optimalAction)
                newValues[state] = v_kplus1
            self.values = newValues.copy()

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
        if self.mdp.isTerminal(state):
            return self.mdp.getReward(state, action, None)

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0.0
        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.getValue(nextState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return None
        qValues = [self.computeQValueFromValues(state, action) for action in possibleActions]
        return possibleActions[qValues.index(max(qValues))]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        cycles = 0
        for i in range(0, self.iterations):
            state = states[i % len(states)]
            optimalAction = self.computeActionFromValues(state)
            v_kplus1 = self.computeQValueFromValues(state, optimalAction)
            self.values[state] = v_kplus1
            # if (i % (len(states) - 1) == 0) and i > 0: cycles += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        # dictionary where keys are states and values are sets of predecessors
        statesAndPredecessors = {}

        # initializing all of the states and their predecessors as sets
        for state in states:
            statesAndPredecessors[state] = set()

        # computing the predecessors
        for state in states:
            possibleActions = self.mdp.getPossibleActions(state)
            for action in possibleActions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action) # list of (nextState, prob) pairs
                for transition in transitions:
                    statesAndPredecessors[transition[0]].add(state)

        prioQueue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                maxQVal = max([self.computeQValueFromValues(state, action) for action in possibleActions])
                diff = abs(self.values[state] - maxQVal)
                prioQueue.push(state, -diff)

        for i in range(0, self.iterations):
            if prioQueue.isEmpty(): break
            state = prioQueue.pop()

            # updating the state's value in self.values
            optimalAction = self.computeActionFromValues(state)
            v_kplus1 = self.computeQValueFromValues(state, optimalAction)
            self.values[state] = v_kplus1

            for predecessor in statesAndPredecessors[state]:
                possibleActions = self.mdp.getPossibleActions(predecessor)
                maxQVal = max([self.computeQValueFromValues(predecessor, action) for action in possibleActions])
                diff = abs(self.values[predecessor] - maxQVal)
                if diff > self.theta: prioQueue.update(predecessor, -diff)
