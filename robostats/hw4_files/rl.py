#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np
import gridworld 

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.1
  This implements value iteration for learning a policy given an environment.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """
  delta = 0
  for round in range(max_iterations):
    vPrev = np.zeros(env.nS)
    for state in env._state_space:
      maxAct = 0
      for action in env._action_space:
        currAct = np.sum(env.P[state][action][0]*(env.P[state][action][2]+gamma*vPrev))
        if currAct > maxAct:
          maxAct = currAct
      vS[state] = maxAct
      if delta < np.abs(vPrev-vS[state]):
        delta = np.abs(vPrev-vS[state])
      vPrev = vS[state]
      if delta < tol:
        return vS, round

def policy_from_value_function(env, value_function, gamma):
  """
  Q3.2.1/Q3.2.2
  This generates a policy given a value function.
  Useful for generating a policy given an optimal value function from value
  iteration.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    value_function: numpy.ndarray
      Optimal value function array of length nS
    gamma: float
      Discount factor, must be in range [0, 1)

  Output:
    numpy.ndarray
    policy: Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
  """
  policy = np.zeros(env.nS)
  for state in env._state_space:
    polActions = np.zeros(env.nA)
    for action in env._action_space:
      polActions[action] = np.sum(env.P[state][action][0]*(env.P[state][action][2]+gamma*value_function))
    policy[state] = np.argmax(polActions)
  return policy

def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.2: BONUS
  This implements policy iteration for learning a policy given an environment.

  You should potentially implement two functions "evaluate_policy" and 
  "improve_policy" which are called as subroutines for this.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """

  ## BONUS QUESTION ##
  ## YOUR CODE HERE ##
  raise NotImplementedError()

def td_zero(env, gamma, policy, alpha):
  """
  Q3.2.2
  This implements TD(0) for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """
  #Vpi = 0
  #for state in env._state_space:

    #update =
    #Vpi = Vpi + alpha*rest

  raise NotImplementedError()

def n_step_td(env, gamma, policy, alpha, n):
  """
  Q3.2.4: BONUS
  This implements n-step TD for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    n: int
      Number of future steps for calculating the return from a state.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """

  ## BONUS QUESTION ##
  ## YOUR CODE HERE ##
  raise NotImplementedError()

def visVals(Vs, width, height):
  fig, ax = plt.subplots()
  visualize = np.reshape(Vs, (width, height))
  img = ax.imshow(visualize)

  for row in range(len(visualize)):
    for col in range(len(visualize)):
      text = ax.text(col, row, round(vis[row, col], 2), ha="center", va="center", color="w")
  plt.title('Values')
  plt.show()

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')

  # Play around with these values if you want!
  gamma = 0.9
  alpha = 0.05
  n = 10
  
  # Q3.2.1
  V_vi, n_iter = value_iteration(env, gamma)
  policy = policy_from_value_function(env, Vs, gamma)

  visVals(V_vi, 8, 8)
  # Q3.2.2: BONUS
  # V_pi, n_iter = policy_iteration(env, gamma)

  # Q3.2.3
  V_td = td_zero(env, gamma, policy, alpha)

  # Q3.2.4: BONUS
  # V_ntd = n_step_td(env, gamma, policy, alpha, n)
