from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

dtype = torch.float32

# the empty grid
nrows = 5
ncols = 10

# ncats is the number of state categories
ncats = 5

# map state categories to states
# want m s.t. r %*% m = reward function
# first, just a map of the indexes
# grid_map = np.array([
# 	[0, 0, 1, 1, 0, 0, 0, 2, 2, 4],
# 	[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
# 	[0, 0, 0, 0, 3, 3, 3, 3, 0, 0]])
grid_map = np.array([
	[0, 1, 1, 1, 0, 0, 0, 2, 2, 4],
	[0, 1, 1, 1, 0, 3, 0, 0, 2, 0],
	[0, 1, 1, 0, 0, 3, 3, 0, 2, 0],
	[0, 1, 1, 0, 3, 3, 3, 0, 2, 0],
	[0, 0, 0, 0, 3, 3, 3, 0, 0, 0]])
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
	beta = 10.0

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
    print("Standard Deviation Violation Count: " + str(round(np.std(violation_map), 3)))
    print("Average Iteration Count: " + str(np.mean(iteration_map)))
    print("Standard Deviation Iteration Count: " + str(round(np.std(iteration_map), 3)))
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


def plotpolicy(pi, startr=0, startc=0, do_print=True):
  grid = []
  count = 0
  for r in range(nrows):
    line = []
    for c in range(ncols):
      line += [6]
    grid += [line]
  findpol(grid, pi, startr, startc)
  # if grid[0][ncols-1] != 4:
  #   count = -1
  # else:
  #   for r in range(nrows):
  #     for c in range(ncols):
  #       if grid[r][c] < 4:
  #         count += 1

  if do_print:
    for r in range(nrows):
      line = ""
      for c in range(ncols):
        line += '^>v<x? '[grid[r][c]]
      print(line)
    # print("Iteration Count: " + str(count))
  return count
    


# all obstacles are lava
# r = np.array([0, -10, -10, -10, 1])

# kids ward is fine to go through
# r = np.array([-1, -1, -20, -20, 10])

# get to goal as quick as possible
# r = np.array([-1, 0, 0, 0, 1])

r = np.array([0.5, -1, -1, -1, 1])
rk = torch.tensor(r,dtype=dtype,requires_grad=False)
piout, Qout = forward(rk)
plotpolicy(piout, 0, 0)
policyViolations(piout)

########### Learning reward function ################


#  0      1     2     3     4
# up, right, down, left, stay

# quick as possible
# trajacts = [1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4]

# scared of everything
# trajacts = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4]

# #kid zone is fine
# trajacts = [2,1,1,1,1,1,1,1,1,1,0,4,4,4,4,4,4]

# kid zone is fine / down
# trajacts = [2,1,1,1,2,1,1,1,1,1,1,0,0,4,4,4,4,4,4]

# goes through yellow
# trajacts = [2,2,2,1,1,1,1,1,1,1,1,1,0,0,0,4,4,4,4,4,4]

# avoid orange
# trajacts = [1,1,1,1,1,1,2,1,1,1,0,4,4,4,4,4]

#optimized
# trajacts = [2,1,1,1,0,4,4,4,4,4]
# trajacts1 = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4]
# trajacts2 = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4]
# trajacts3 = [2,1,1,0,4,4,4,4,4]
# trajacts4 = [2,1,1,1,0,4,4,4,4,4]
# trajacts5 = [2,1,1,1,0,4,4,4,4,4]
# trajacts6 = [2,1,1,1,0,4,4,4,4,4]
# trajacts7 = [2,1,1,1,0,4,4,4,4,4]
# trajacts8 = [2,1,1,0,4,4,4,4,4]
# trajacts9 = [2,1,1,0,4,4,4,4,4]
# trajacts10 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4]

# trajcoords = reduce(
#     (lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts1, [[0, 1]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts2, [[0, 0]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts3, [[0, 7]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts4, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts5, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts6, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts7, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts8, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts9, [[0, 6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts10, [[4, 0]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts10, [[4, 0]])

# trajacts += trajacts1
# trajacts += trajacts2
# trajacts += trajacts3
# trajacts += trajacts4
# trajacts += trajacts5
# trajacts += trajacts6
# trajacts += trajacts7
# trajacts += trajacts8
# trajacts += trajacts9
# trajacts += trajacts10
# trajacts += trajacts10


# test
# trajacts = [2,2,2,2,1,1,1,0,3,3,0,1,1,1,1,1,1,1,2,2,1,0,0,0,0,4,4,4,4,4]

# trajacts = [4,4,4,4,4]

# complicated map trajectories
# trajacts = [2, 2, 2, 2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4]
# trajacts1 = [2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4]


## optimized trajectory
trajacts = [2, 2, 2, 2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 4, 4, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4]
trajcoords = reduce(
    (lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [[0, 0]])
# trajacts += trajacts1
# trajacts += trajacts2
# trajacts += trajacts3

# trajcoords = reduce(
#     (lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [[0, 1]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts1, [[0, 6]])


## open hall map trajectories

# trajacts = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4]
# trajacts1 = [2, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4]
# trajacts2 = [ 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4]
# trajcoords = reduce(
#     (lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [[0, 0]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts1, [[1,6]])
# trajcoords += reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts2, [[3, 2]])
# trajacts += trajacts1
# trajacts += trajacts2
# trajacts += trajacts3


#  0      1     2     3     4
# up, right, down, left, stay


# trajacts += trajacts3

# print(trajacts)
# print(trajcoords)


learning_rate = 0.001
iterations = 200

# initial random guess on r
# r = np.random.rand(5)*2-1
r = np.array([ 0, 0, 0, 0, 0 ])
rk = torch.tensor(r, dtype=dtype, requires_grad=True)
lossList = []
pvList = []
iterList = []

for iter in range(iterations):
  piout, Qout = forward(rk)
  # print("my pi:",piout[0][0])

  loss = 0
  for i in range(len(trajacts)):
    acti = trajacts[i]
    state = trajcoords[i]
    loss += -torch.log(piout[state[0]*ncols+state[1]][acti])

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
    rk -= learning_rate * grads_value
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


  if iter % 10 == 0:
    print(loss, rk)
    lossList.append(loss.item())
    iter, viol = policyViolations(piout)
    pvList.append(viol)
    # print(piout)
    # print(Qout)
    iterList.append(iter)
  else:
    lossList.append(loss.item())
    iter, viol = policyViolations(piout, do_print=False)
    pvList.append(viol)
    iterList.append(iter)

plt.plot(lossList)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.suptitle('Action-Feedback Loss Graph')
plt.show()

plt.plot(pvList)
plt.xlabel('Iteration')
plt.ylabel('Policy Violation Average')
plt.suptitle('Action-Feedback Average Violation Graph')
plt.show()

plt.plot(iterList)
plt.xlabel('Iteration')
plt.ylabel('Steps to Goal')
plt.suptitle('Action-Feedback Average Steps Graph')
plt.show()