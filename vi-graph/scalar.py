from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import time
import random
from matplotlib.widgets import Button

dtype = torch.float32

# the empty grid
nrows = 5
ncols = 10

# ncats is the number of state categories
ncats = 5

# map state categories to states
# want m s.t. r %*% m = reward function
# first, just a map of the indexes
grid_map = np.array([
	[0, 0, 1, 1, 0, 0, 0, 2, 2, 4],
	[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
	[0, 0, 0, 0, 3, 3, 3, 3, 0, 0]])
# grid_map = np.array([
# 	[0, 1, 1, 1, 0, 0, 0, 2, 2, 4],
# 	[0, 1, 1, 1, 0, 3, 0, 0, 2, 0],
# 	[0, 1, 1, 0, 0, 3, 3, 0, 2, 0],
# 	[0, 1, 1, 0, 3, 3, 3, 0, 2, 0],
# 	[0, 0, 0, 0, 3, 3, 3, 0, 0, 0]])
# grid_map = np.array([
# 	[0, 1, 1, 1, 3, 3, 3, 2, 2, 4],
# 	[0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
# 	[0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
# 	[0, 1, 1, 1, 3, 3, 3, 2, 2, 0],
# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
colors = ["white", "blue", "orange", "yellow", "green"]


# sns.heatmap(grid_map, cmap=sns.xkcd_palette(colors), yticklabels=False, xticklabels=False,
# 	annot=False, cbar=False, annot_kws={"size": 30}, linewidths=1, linecolor="gray")
# plt.show()

# r is the rewards for the different location categories
# r = np.array([0, -1, -1, -1, 10])
# binarized form of the map (rows,columns,category)
matmap = np.zeros((nrows, ncols, ncats))
for i in range(nrows):
	for j in range(ncols):
		for k in range(ncats):
			matmap[i, j, k] = 0+(grid_map[i, j] == k)


def clip(v, min, max):
	if v < min: v = min
	if v > max-1: v = max-1
	return(v)


acts = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
nacts = len(acts)
# matrans goes (action,state,state), where we have turned (row,col) into state=row*ncols+col (flattened)
mattrans = np.zeros((nacts, nrows*ncols, nrows*ncols))
for acti in range(nacts):
	act = acts[acti]
	for i1 in range(nrows):
		for j1 in range(ncols):
			inext = clip(i1 + act[0], 0, nrows)
			jnext = clip(j1 + act[1], 0, ncols)
			for i2 in range(nrows):
				for j2 in range(ncols):
					mattrans[acti, i1*ncols+j1, i2*ncols+j2] = 0 + \
					    ((i2 == inext) and (j2 == jnext))

matmap = torch.tensor(matmap, dtype=dtype, requires_grad=False)
mattrans = torch.tensor(mattrans, dtype=dtype, requires_grad=False)


def forward(rk):
	# r starts as [n] categories. we will inflate this to [rows,cols,n] to multiple with matmap and sum across category
	# print("rk shape:",rk.shape)
	new_rk = rk.unsqueeze(0)
	new_rk = new_rk.unsqueeze(0)
	new_rk = new_rk.expand(nrows, ncols, len(r))
	# print("rk shape:",new_rk.shape)
	# print("matmap shape:",matmap.shape)
	rfk = torch.mul(matmap, new_rk)
	# print("rfk shape:",rfk.shape)
	rfk = rfk.sum(axis=-1)
	# print("rfk shape:",rfk.shape)
	rffk = rfk.view(nrows*ncols)
	# print("rffk shape:",rffk.shape)
	# print("requires_grad rffk:",rffk.requires_grad)

	# initialize the value function to be the reward function, required_grad should be false
	v = rffk.detach().clone()
	# print("v shape:",v.shape)
	gamma = 0.90
	beta = 1

	# inflate v to be able to multiply mattrans
	# print("mattrans shape:",mattrans.shape)
	v = v.unsqueeze(0)
	v = v.expand(nrows*ncols, nrows*ncols)

	for _ in range(50):
		q0 = torch.mul(mattrans[0], v)
		q0 = q0.sum(axis=-1)
		q1 = torch.mul(mattrans[1], v)
		q1 = q1.sum(axis=-1)
		q2 = torch.mul(mattrans[2], v)
		q2 = q2.sum(axis=-1)
		q3 = torch.mul(mattrans[3], v)
		q3 = q3.sum(axis=-1)
		q4 = torch.mul(mattrans[4], v)
		q4 = q4.sum(axis=-1)

		# print("q0 shape:",q0.shape)

		Q = torch.stack((q0, q1, q2, q3, q4), axis=-1)
		# print("Q:",Q.shape)
		pytorch_sm = nn.Softmax(dim=1)
		pi = pytorch_sm(beta*Q)
		# print("pi:",pi.shape)
		# print(pi[:][-1])

		next_q = (Q*pi).sum(axis=1)
		# print("next q:",next_q.shape)
		v = rffk + gamma * next_q
		# print("new v:",v.shape)
		# print("requires_grad maybe:",v.requires_grad)

	return(pi, Q)


def policyViolations(pi):
  violation_map = np.zeros((nrows, ncols))
  for i in range(nrows):
    for j in range(ncols):
      grid = []
      for r in range(nrows):
        line = []
        for c in range(ncols):
          line += [6]
        grid += [line]
      violation_map[i][j] = stateViolations(grid, pi, i, j)
  print("Policy Violation Map:")
  print(violation_map)
  print("Average Policy Violation Count: " + str(np.mean(violation_map)))
  return np.mean(violation_map)

# returns number of violations in a state


def stateViolations(grid, pi, r, c):
  if grid[r][c] != 6: return 0
  maxprob = max(pi[r*ncols+c, :])
  a = 6
  for ana in range(5):
    if pi[r*ncols+c, ana] == maxprob: a = ana
  grid[r][c] = a
  r += acts[a][0]
  c += acts[a][1]
  tile_type = grid_map[r][c]
  if tile_type == 1 or tile_type == 2 or tile_type == 3:
	  return stateViolations(grid, pi, r, c) + 1
  return stateViolations(grid, pi, r, c)


def findpol(grid, pi, r, c):
  if grid[r][c] != 6: return
  maxprob = max(pi[r*ncols+c, :])
  a = 6
  for ana in range(5):
    if pi[r*ncols+c, ana] == maxprob: a = ana
  grid[r][c] = a
  r += acts[a][0]
  c += acts[a][1]
  findpol(grid, pi, r, c)


def plotpolicy(pi, startr=0, startc=0):
  grid = []
  for r in range(nrows):
    line = []
    for c in range(ncols):
      line += [6]
    grid += [line]
  findpol(grid, pi, startr, startc)
  for r in range(nrows):
    line = ""
    for c in range(ncols):
      line += '^>v<x? '[grid[r][c]]
    print(line)

def printGrid(r, c, action):
	move_grid = np.empty(shape=(nrows, ncols), dtype='str')
	move_grid[r][c] = '^>v<x'[action]
	sns.heatmap(grid_map, cmap=sns.xkcd_palette(colors), yticklabels=False, xticklabels=False, cbar=False, annot=move_grid, fmt="", annot_kws={"size": 30}, linewidths=1, linecolor="gray")
	# axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
	# axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
	# bnext = Button(axnext, 'Next')
	# bnext.on_clicked(lambda x: print("next"))
	# bprev = Button(axprev, 'Previous')
	# bprev.on_clicked(lambda x: print("previous"))
	plt.show()

def chooseAction(pi, r, c):
	epsilon = 0.25
	action_prob = pi[r*ncols+c].cpu().detach().numpy()
	if r == 0:
		action_prob[0] = 0
	if c == ncols - 1:
		action_prob[1] = 0
	if r == nrows - 1:
		action_prob[2] = 0
	if c == 0:
		action_prob[3] = 0

	action_prob = action_prob / np.sum(action_prob)
	print("Action Probabilities (Up, Right, Down, Left, Stay): ")
	print(np.round(action_prob, 3))

	r = random.uniform(0, 1)
	# print("Random Number: " + str(r))

	# if r < epsilon:
	# 	permitable_actions = np.nonzero(action_prob)[0]
	# 	choice = np.random.choice(permitable_actions, 1)[0]
	# 	print("Picking a random action...")
	# 	print(choice)
	# 	return choice


	print("Picking from probabilities...")
	cp = [0, np.cumsum(action_prob)]
	choice = np.random.choice(5, 1, p=action_prob)[0]
	# print(choice)
	# choice = np.argmax(action_prob)
	print()
	return choice



# all obstacles are lava
# r = np.array([0, -10, -10, -10, 1])

# kids ward is fine to go through
# r = np.array([-1, -1, -20, -20, 10])

# get to goal as quick as possible
# r = np.array([-1, 0, 0, 0, 1])

# rk = torch.tensor(r,dtype=dtype,requires_grad=False)
# piout, Qout = forward(rk)
# policyViolations(piout)




learning_rate = 0.01

# initial random guess on r
# r = np.random.rand(5)*2-1
r = np.array([ 0, 0, 0, 0, 1 ])
rk = torch.tensor(r, dtype=dtype, requires_grad=True)
lossList = []
pvList = []

row_start = 0
column_start = 0
row_dest = 0
column_dest = 9

row_state = row_start
column_state = column_start

for iter in range(201):
	if row_state == row_dest and column_state == column_dest: 
		row_state = row_start
		column_state = column_start
		print("Robot Reached Goal... Resetting Position to (" + str(row_state) + ", " + str(column_state) + ")")
		time.sleep(2.5)

	piout, Qout = forward(rk)
	action = chooseAction(piout, row_state, column_state)
	# scalar_feedback = input("Give the robot feedback on it's action (-1, 0, 1): ")
	# for i in range(len(trajacts)):
	# 	acti = trajacts[i]
	# 	state = trajcoords[i]
	loss = 0
	pi = piout[row_state*ncols+column_state][action]
	# print(pi)

	loss = -torch.log(pi)
	# loss = -torch.log(pi * scalar / pi)



	# print("does rk need grad",rk.requires_grad)
	loss.backward()
	# print("The loss is:",loss)
	with torch.no_grad():
		grads_value = rk.grad
		# print("our grads:",grads_value)

		# print("grad values are:",grads_value)
		# print("old rk:",rk)
		# print("grads value:",grads_value)
		# print("intermediate:",learning_rate * grads_value)
		printGrid(row_state, column_state, action)
		scalar_feedback = float(input("Give the robot scalar feedback on it's action: "))
		scale = scalar_feedback / pi.item()
		min_scale = -2.5
		max_scale = 2.5
		if scale > max_scale:
			scale = max_scale
		if scale < min_scale:
			scale = min_scale
		print("rk Scale: " + str(scale))
		# rk -= learning_rate * grads_value
		rk -= (learning_rate * grads_value) * scale
		# multiply by the scalar number divided by the negative log
		# print("new rk:",rk)
		# print(rk)
		# rk_mean = torch.mean(rk)
		# rk_std = torch.std(rk)
		# rk.copy_((rk - rk_mean) / rk_std)
		rk_min = torch.min(rk)
		rk_max = torch.max(rk)
		# rk.copy_(2 * ((rk - rk_min) / (rk_max - rk_min)) - 1)
		rk.grad.zero_()

	# rk -= learning_rate * grads_value
	# [-0.0377, -0.0876, -0.0304, -0.3253,  1.3532]
	# [-0.0221, -0.1680, -0.0517, -0.3193,  1.3863]

	# [-0.0629, -0.0807, -0.0205, -0.0030,  0.9147]
	# [-0.2615, -0.2069, -0.0045,  0.0909,  1.0278]
	lossList.append(loss.item())
	print("Loss, Reward (White, Blue, Orange, Yellow, Green): ")
	print(loss, rk)

	if action == 0:
		row_state -= 1
	if action == 1:
		column_state += 1
	if action == 2:
		row_state += 1
	if action == 3:
		column_state -= 1

	# plotpolicy(piout,0,0)
	# pvList.append(policyViolations(piout))

# plt.plot(lossList)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.suptitle('Action-Feedback Loss Graph')
# plt.show()
# plt.plot(pvList)
# plt.xlabel('Iteration')
# plt.ylabel('Policy Violation Average')
# plt.suptitle('Action-Feedback Average Violation Graph')
# plt.show()
