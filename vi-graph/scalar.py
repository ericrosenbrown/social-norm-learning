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
	beta = 1.25

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


def policyViolations(pi, do_print=True):
  violation_map = np.zeros((nrows, ncols))
  iteration_map = np.zeros((nrows, ncols))
  for i in range(nrows):
    for j in range(ncols):
      grid = []
      for r in range(nrows):
        line = []
        for c in range(ncols):
          line += [6]
        grid += [line]
      iter, viol = stateViolations(grid, pi, i, j)
      violation_map[i][j] = viol
      iteration_map[i][j] = iter
  if do_print:
    print("Policy Violation Map:")
    print(violation_map)
    print("Iteration Map:")
    print(iteration_map)
    print("Average Policy Violation Count: " + str(np.mean(violation_map)))
    # print("Standard Deviation Violation Count: " + str(round(np.std(violation_map), 3)))
    print("Average Iteration Count: " + str(np.mean(iteration_map)))
    # print("Standard Deviation Iteration Count: " + str(round(np.std(iteration_map), 3)))
  return (np.mean(iteration_map), np.mean(violation_map))

# returns number of violations in a state


def stateViolations(grid, pi, r, c):
  if grid[r][c] != 6: return (0, 0)
  maxprob = max(pi[r*ncols+c, :])
  a = 6
  for ana in range(5):
    if pi[r*ncols+c, ana] == maxprob: a = ana
  grid[r][c] = a
  r += acts[a][0]
  c += acts[a][1]
  iter, viol = stateViolations(grid, pi, r, c)
  if grid[r][c] < 4:
      iter += 1
  tile_type = grid_map[r][c]
  if tile_type == 1 or tile_type == 2 or tile_type == 3:
	  viol += 1
  return (iter, viol)



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

# def findPaths(pi, startr=0, startc=0):
#   grid = []
#   for r in range(nrows):
#     line = []
#     for c in range(ncols):
#       line += [6]
#     grid += [line]
#   findpol(grid, pi, startr, startc)
#   for r in range(nrows):
#     for c in range(ncols):
#       grid[r][c] = '^>v<x? '[grid[r][c]]
#   return grid	

def findPaths(pi, startr=0, startc=0):
  grid1 = []
  grid2 = []
  grid3 = []
  grid4 = []
  for r in range(nrows):
    line1 = []
    line2 = []
    line3 = []
    line4 = []
    for c in range(ncols):
      line1 += [6]
      line2 += [6]
      line3 += [6]
      line4 += [6]
    grid1 += [line1]
    grid2 += [line2]
    grid3 += [line3]
    grid4 += [line4]
  if startr != 0:
    findpol(grid1, pi, startr-1, startc)
  if startc != 0:
    findpol(grid2, pi, startr, startc-1)
  if startr != nrows-1:
    findpol(grid3, pi, startr+1, startc)
  if startc != ncols-1:
    findpol(grid4, pi, startr, startc+1)
  for r in range(nrows):
    for c in range(ncols):
      grid1[r][c] = '^>v<x? '[grid1[r][c]]
      if grid2[r][c] != 6:
        grid1[r][c] = '^>v<x? '[grid2[r][c]]
      if grid3[r][c] != 6:
        grid1[r][c] = '^>v<x? '[grid3[r][c]]
      if grid4[r][c] != 6:
        grid1[r][c] = '^>v<x? '[grid4[r][c]]
  return grid1



def set_scalar(input):
  global scalar 
  scalar = input
  # print(scalar)
  plt.close()

def printGrid(r, c, nr, nc, action, piout):
  plt.close() 
  # plt.ion()
  move_grid = np.empty(shape=(nrows, ncols), dtype='str')
  move_grid[r][c] = '^>v<x'[action]
  # move_grid[nr][nr] = "o"

  next_moves_grid = findPaths(piout, r, c)
  next_moves_grid[r][c] = "+"

  fig, (ax1, ax2) = plt.subplots(2, figsize=[6, 6])

  sns.heatmap(grid_map, ax=ax1, cmap=sns.xkcd_palette(colors), yticklabels=False, xticklabels=False, cbar=False, annot=move_grid, fmt="", annot_kws={"size": 30}, linewidths=1, linecolor="gray")
  sns.heatmap(grid_map, ax=ax2, cmap=sns.xkcd_palette(colors), yticklabels=False, xticklabels=False, cbar=False, annot=next_moves_grid, fmt="", annot_kws={"size": 30}, linewidths=1, linecolor="gray")
  ax1.set_title('Robot Action')
  ax2.set_title('Robot Most Likely Pathways')
  plt.draw()
  
	

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


	# print("Picking from probabilities...")
	# choice = np.random.choice(5, 1, p=action_prob)[0]
	# print(choice)
	choice = np.argmax(action_prob)
	return choice



# all obstacles are lava
# r = np.array([0, -10, -10, -10, 1])

# kids ward is fine to go through
# r = np.array([-1, -1, -20, -20, 10])

# get to goal as quick as possible
# r = np.array([-1, 0, 0, 0, 1])

# r = np.array([0.0479, -.0492, -.1264, -.0389, 0.6370])
# rk = torch.tensor(r,dtype=dtype,requires_grad=False)
# piout, Qout = forward(rk)
# policyViolations(piout)
# printGrid(0,0,0,0,4,piout)
# plt.show()



learning_rate = 0.0055

# initial random guess on r
# r = np.random.rand(5)*2-1
r = np.array([0, 0, 0, 0, 1])
# r = np.array([0, -10, -10, -10, 10])
rk = torch.tensor(r, dtype=dtype, requires_grad=True)
lossList = []
pvList = []
iterList = []

row_start = 0
column_start = 0
row_dest = 0
column_dest = 9

p = 0
#basic map
# p_rows = [0,4,0,4,0,0]
# p_cols = [6,3,1,9,4,0]
# p_rows = [0,4,0,0,4,0,4,0]
# p_cols = [6,3,1,5,9,4,0,0]
#complicated map
p_rows = [0, 4, 2, 4]
p_cols = [0, 7, 4, 3]
# #open hall map
# p_rows = [0,0,0,2,2]
# p_cols = [6,7,0,3,6]

row_state = row_start
column_state = column_start
reset_count = 0
resets = 20
while(reset_count < resets):
  if row_state == row_dest and column_state == column_dest: 
    if p > (len(p_rows) - 1):
      p = 0
    row_state = p_rows[p]
    column_state = p_cols[p]
    if reset_count == resets:
      print("Finished Iterations!")
      break
    print("Robot Reached Goal... Resetting Position to (" + str(row_state) + ", " + str(column_state) + ")")
    print("Loss, Reward (White, Blue, Orange, Yellow, Green): ")
    print(loss, rk)
    p += 1
    time.sleep(2.5)
    reset_count += 1

  piout, Qout = forward(rk)
  if False:
    print(loss, rk)
    # lossList.append(loss.item())
    iter, viol = policyViolations(piout)
    pvList.append(viol)
    iterList.append(iter)
  else:
    iter, viol = policyViolations(piout)
    pvList.append(viol)
    iterList.append(iter)


  action = chooseAction(piout, row_state, column_state)
  # scalar_feedback = input("Give the robot feedback on it's action (-1, 0, 1): ")
  # for i in range(len(trajacts)):
  # 	acti = trajacts[i]
  # 	state = trajcoords[i]
  loss = 0
  pi_action_state = piout[row_state*ncols+column_state][action]
  # print(pi)

  loss = -torch.log(pi_action_state)
  # loss = -torch.log(pi * scalar / pi_action_state)



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
    nr = row_state
    nc = column_state
    if action == 0:
      nr -= 1
    if action == 1:
      nc += 1
    if action == 2:
      nr += 1
    if action == 3:
      nc -= 1

    printGrid(row_state, column_state, nr, nc, action, piout)

    # while(not has_feedback):
    # 	time.sleep(1)
    # has_feedback = False
    # print(scalar)
    scalar_feedback = 0
    plt.text(0, 5.65, "Give the robot feed-")
    plt.text(0, 6.15, "back on it's action:")
    # ax0 = plt.axes([0.38, 0.02, 0.1, 0.075])
    ax1 = plt.axes([0.49, 0.02, 0.1, 0.075])
    ax2 = plt.axes([0.6, 0.02, 0.1, 0.075])
    ax3 = plt.axes([0.71, 0.02, 0.1, 0.075])
    # ax4 = plt.axes([0.82, 0.02, 0.1, 0.075])
    # b0 = Button(ax0, '-2')
    # b0.on_clicked(lambda x: set_scalar(-2))
    b1 = Button(ax1, '-1')
    b1.on_clicked(lambda x: set_scalar(-1))
    b2 = Button(ax2, '0')
    b2.on_clicked(lambda x: set_scalar(0))
    b3 = Button(ax3, '1')
    b3.on_clicked(lambda x: set_scalar(1))
    # b4 = Button(ax4, '2')
    # b4.on_clicked(lambda x: set_scalar(2))
    plt.draw()
    # plt.ion()
    plt.show()
    try: 
      # input_text = float(input("Give the robot scalar feedback on it's action: "))
      # scalar_feedback = input_text
      scalar_feedback = scalar
      # print(scalar_feedback)
      
    except:
      print("Non-Numerical Value Given. Feedback will be 0 for this action")
      time.sleep(1.5)
    scale = clip(scalar_feedback / pi_action_state.item(), -1, 2)
    # scale = scalar_feedback / pi_action_state.item()
    # print("rk Scale: " + str(scale))
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
    
        # lossList.append(loss.item())
    # print("Loss, Reward (White, Blue, Orange, Yellow, Green): ")
    # print(loss, rk)    


  # rk -= learning_rate * grads_value
  # [-0.0377, -0.0876, -0.0304, -0.3253,  1.3532]
  # [-0.0221, -0.1680, -0.0517, -0.3193,  1.3863]

  # [-0.0629, -0.0807, -0.0205, -0.0030,  0.9147]
  # [-0.2615, -0.2069, -0.0045,  0.0909,  1.0278]
  row_state = nr
  column_state = nc


  # if iter % 10 == 0:
  
policyViolations(piout, do_print=True)
# plt.close()
# plt.plot(lossList)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.suptitle('Scalar-Feedback Loss Graph')
# plt.show()

plt.close()
plt.plot(pvList)
plt.xlabel('Iteration')
plt.ylabel('Policy Violation Average')
plt.suptitle('Scalar-Feedback Average Violation Graph')
plt.show()

plt.close()
plt.plot(iterList)
plt.xlabel('Iteration')
plt.ylabel('Steps to Goal')
plt.suptitle('Scalar-Feedback Average Steps Graph')
plt.show()