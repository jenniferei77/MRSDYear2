#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# IRL questions:
# Fill in the various functions in this file for Q3.3 on the project.

import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt

import gridworld
import rl

import pdb

def visVals(Vs, width, height):
  fig, ax = plt.subplots()
  visualize = np.reshape(Vs, (width, height))
  img = ax.imshow(visualize)

  for row in range(len(visualize)):
    for col in range(len(visualize)):
      text = ax.text(col, row, round(visualize[row, col], 2), ha="center", va="center", color="w")
  plt.title('Values')
  plt.show()

def irl_lp(policy, T_probs, discount, R_max, l1):
  T_probs = np.asarray(T_probs)
  nS, nA, _ = T_probs.shape

  G = np.zeros([2*nS*(nA+1), 3*nS])
  h = np.zeros([2*nS*(nA+1)])
  c = np.zeros([3*nS])
  R = np.full([nS], R_max)

  G_block = []
  G_col2 = G_col3 = np.zeros([2*nS*(nA+1), nS])
  for state in range(nS):
    aStar = policy[state]
    inv_part = np.linalg.inv(np.identity(nS)-discount*T_probs[:,aStar,:])
    action_count = 0
    for action in range(nA):
      if action != aStar:
        prob_diff = T_probs[state, aStar, :] - T_probs[state, action, :]
        G_block.append(np.dot(-prob_diff, inv_part))
        G_col2[nS*(nA-1)+state*(nA-1)+action_count, state] = 1
        action_count += 1

  G_col1 = np.vstack([G_block, G_block, -np.identity(nS), np.identity(nS), -np.identity(nS), np.identity(nS)])
  G_col3 = np.vstack([np.zeros([np.size(G_block,0)*2,nS]), -np.identity(nS), -np.identity(nS), np.zeros([nS*2,nS])])
  G = np.hstack([G_col1, G_col2, G_col3])

  #h[2*nS*(nA-1):2*nS*(nA-1)+nS] = R
  #h[2*nS*(nA-1)+nS:] = 1
  h[:3*nS] = R_max

  c[nS:nS*2] = -1
  c[nS*2:] = l1

  # You shouldn't need to touch this part.
  c = cvx.matrix(c)
  G = cvx.matrix(G)
  h = cvx.matrix(h)
  sol = cvx.solvers.lp(c, G, h)

  R = np.asarray(sol["x"][:nS]).squeeze()

  return R

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')

  # Generate policy from Q3.2.1
  gamma = 0.75
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)

  T = env.generateTransitionMatrices()

  # Q3.3.5
  # Set R_max and l1 as you want.
  R_max = 1
  l1 = 0.8
  R = irl_lp(policy, T, gamma, R_max, l1)
  visVals(R, 8, 8)
   

  # You can test out your R by re-running VI with your new rewards as follows:
  env_irl = gridworld.GridWorld(map_name='8x8', R=R)
  Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
  policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)
  visVals(policy_irl, 8, 8)