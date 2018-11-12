#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np
import gridworld 
import pdb
import matplotlib.pyplot as plt

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  delta = 0
  vS = np.zeros(env.nS)
  for r in range(max_iterations):
    for state in env._state_space:
      maxAct = -np.inf
      currAct = 0
      for action in env._action_space:
      	for future_state in env.P[state][action]:
      		reward = future_state[2]
      		pstate = future_state[1]
      		prob = future_state[0]
        	currAct += prob*(reward+gamma*vS[pstate])
        if currAct > maxAct:
          maxAct = currAct
      if delta < np.abs(vS[state]-maxAct):
        delta = np.abs(vS[state]-maxAct)
      vS[state] = maxAct
    if delta < tol:
      return vS, r
  return vS, r



def policy_from_value_function(env, value_function, gamma):
  policy = np.zeros(env.nS)
  for state in env._state_space:
    polActions = np.zeros(env.nA)

    for action in env._action_space:
    	currAct = 0
    	for future_state in env.P[state][action]:
      		currAct += future_state[0]*(future_state[2]+gamma*value_function[state])
      	polActions[action] = currAct
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
  Vpi = np.zeros(env.nS)
  max_iterations = int(1e3)
  for n in range(max_iterations):
  	for state in env._state_space:	
	  for future in env.P[state][policy[state]]:
	  	reward = future[2] 
		future_state = future[1]
		Vs = Vpi[state]
		Vsp = Vpi[future_state]
		Vpi[state] = Vs + alpha*(reward+gamma*Vsp - Vs) 
  return Vpi

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
      text = ax.text(col, row, round(visualize[row, col], 2), ha="center", va="center", color="w")
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
  print V_vi

  #visVals(V_vi, 8, 8)

  policy = policy_from_value_function(env, V_vi, gamma)
  print policy
  #visVals(policy, 8, 8)

  # Q3.2.2: BONUS
  # V_pi, n_iter = policy_iteration(env, gamma)

  # Q3.2.3
  V_td = td_zero(env, gamma, policy, alpha)
  print V_td
  visVals(V_td, 8, 8)

#V_vi:
# [  0.00000000e+00  -1.45142395e-03  -5.93593830e-03  -1.05026500e-01
#   -1.80062106e-01  -2.39274360e-01  -2.76530584e-01  -2.92296484e-01
#    0.00000000e+00  -4.99934917e-03  -1.75431618e-02  -3.49886066e-01
#   -4.10160922e-01  -4.64043028e-01  -4.73945515e-01  -4.53737831e-01
#    0.00000000e+00  -2.07679057e-02  -6.20349713e-02  -3.03825249e-01
#   -9.43103192e-01  -1.00281725e+00  -9.01807870e-01  -7.76425069e-01
#    0.00000000e+00  -8.73024539e-02  -2.16632010e-01  -8.76378433e-01
#   -9.51536098e-01  -9.95605051e-01  -1.52845496e+00  -1.19342451e+00
#    0.00000000e+00  -3.67243001e-01  -7.26169056e-01  -9.35703765e-01
#   -1.53318927e+00  -1.51902189e+00  -1.67789294e+00  -1.47077396e+00
#    0.00000000e+00  -4.33777549e-01  -1.16518890e+00  -1.82996686e+00
#   -1.76900866e+00  -1.57811369e+00  -1.77969214e+00  -1.98756275e+00
#    0.00000000e+00  -4.49546105e-01  -1.36267098e+00  -1.53377112e+00
#   -1.55802664e+00  -1.95679941e+00  -1.96451140e+00  -1.58123175e+00
#    0.00000000e+00  -4.53094031e-01  -6.58701012e-01  -1.15034059e+00
#   -2.08801195e+00  -1.78049709e+00  -1.92675968e+00   0.00000000e+00]
#policy:
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  3.  0.  0.  0.  0.  0.  0.
#   0.  0.  2.  3.  0.  0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  3.  0.  0.
#   2.  1.  3.  0.  0.  0.  2.  1.  3.  0.  3.  2.  0.  1.  1.  3.  0.  0.
#   1.  1.  0.  1.  0.  3.  1.  0.  1.  0.]
#V_td:
# [ 0.         -0.00494071 -0.02332233 -0.03518617 -0.0641044  -0.06884807
#  -0.06321819 -0.05453904  0.         -0.0112003  -0.04817675 -0.05760889
#  -0.11189424 -0.09517557 -0.07790647 -0.0635797   0.         -0.03103975
#  -0.12100483 -0.42444721 -0.24205992 -0.13321876 -0.09944902 -0.0782797   0.
#  -0.0882705  -0.31055081 -1.17831523 -0.58586529 -0.3473884  -0.11832973
#  -0.0962459   0.         -0.25175508 -0.79120971 -0.9426289  -0.51900893
#  -0.42422627 -0.20405195 -0.12198537  0.         -0.53967901 -0.87714467
#  -1.1725312  -0.72696923 -0.707478   -0.31453943 -0.10591457  0.
#  -0.37216258 -0.96947032 -1.25779201 -0.99761681 -1.16882724 -0.35816122
#  -0.12495856  0.         -0.22113504 -0.50130086 -0.90845244 -1.31089892
#  -1.06274257  0.04700322  0.        ]

  # Q3.2.4: BONUS
  # V_ntd = n_step_td(env, gamma, policy, alpha, n)
