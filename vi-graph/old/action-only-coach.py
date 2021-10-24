import numpy as np
import random
import copy
from numpy.random import choice
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

dtype = torch.float32

# the empty grid
nrows = 5
ncols = 10
 
# ncats is the number of state categories
ncats = 5
 
# map state categories to states
# want m s.t. r %*% m = reward function
# first, just a map of the indexes
grid_map = np.array([ [0,0,1,1,0,0,0,2,2,4],
	[0,0,1,1,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,3,3,3,3,3,3,0,0],
	[0,0,0,0,3,3,3,3,0,0]])

grid_map2 = np.array([ [4,3,3,3,3,0,2,2,0,0],
	[0,3,3,3,3,0,2,2,0,0],
	[0,0,1,1,0,0,2,2,0,0],
	[0,0,1,1,0,0,2,2,0,0],
	[0,0,0,0,0,0,0,0,0,0]])

grid_map3 = np.array([ [0,0,0,0,0,0,0,0,0,0],
	[0,0,2,2,0,3,3,0,0,0],
	[0,0,2,2,4,3,3,0,0,0],
	[0,0,2,2,1,1,1,0,0,0],
	[0,0,0,0,1,1,1,0,0,0]])

all_grids = [grid_map,grid_map2,grid_map3]
state_starts = [[4,3],[4,0],[4,0]]
viz_starts = [[0,0],[4,0],[4,0]]
grid_idx = 0

cur_grid_map = all_grids[grid_idx]

colors = ["white", "blue", "orange", "yellow", "green"]

def viz_map(robox,roboy):
    print("===================================")
    #grid_map[robox][roboy] = 9
    for r in range(len(cur_grid_map)):
        rowstr = ""
        for c in range(len(cur_grid_map[r])):
            if r==robox and c==roboy:
                rowstr += "R"
            else:            
                rowstr += str(cur_grid_map[r][c])
        print(rowstr)

# r is the rewards for the different location categories
#r = np.array([0, -1, -1, -1, 10])
#binarized form of the map (rows,columns,category)
matmap = np.zeros((nrows,ncols,ncats))
for i in range(nrows):
	for j in range(ncols):
		for k in range(ncats):
			matmap[i,j,k] = 0+(cur_grid_map[i,j] == k)
def clip(v,min,max):
	if v < min: v = min
	if v > max-1: v = max-1
	return(v)
 
acts = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
nacts = len(acts)
#matrans goes (action,state,state), where we have turned (row,col) into state=row*ncols+col (flattened)
mattrans = np.zeros((nacts,nrows*ncols,nrows*ncols))
for acti in range(nacts):
	act = acts[acti]
	for i1 in range(nrows):
		for j1 in range(ncols):
			inext = clip(i1 + act[0],0,nrows)
			jnext = clip(j1 + act[1],0,ncols)
			for i2 in range(nrows):
				for j2 in range(ncols):
					mattrans[acti,i1*ncols+j1,i2*ncols+j2] = 0+((i2 == inext) and (j2 == jnext))

matmap = torch.tensor(matmap,dtype=dtype,requires_grad=False)
mattrans = torch.tensor(mattrans,dtype=dtype,requires_grad=False)

def forward(rk):
	#r starts as [n] categories. we will inflate this to [rows,cols,n] to multiple with matmap and sum across category
	#print("rk shape:",rk.shape)
	new_rk = rk.unsqueeze(0)
	new_rk = new_rk.unsqueeze(0)
	new_rk = new_rk.expand(nrows,ncols,len(r))
	#print("rk shape:",new_rk.shape)
	#print("matmap shape:",matmap.shape)
	rfk = torch.mul(matmap,new_rk)
	#print("rfk shape:",rfk.shape)
	rfk = rfk.sum(axis=-1)
	#print("rfk shape:",rfk.shape)
	rffk = rfk.view(nrows*ncols)
	#print("rffk shape:",rffk.shape)
	#print("requires_grad rffk:",rffk.requires_grad)

	#initialize the value function to be the reward function, required_grad should be false
	v = rffk.detach().clone()
	#print("v shape:",v.shape)
	gamma = 0.90
	beta = 10.0

	#inflate v to be able to multiply mattrans
	#print("mattrans shape:",mattrans.shape)
	v = v.unsqueeze(0)
	v = v.expand(nrows*ncols,nrows*ncols)

	for _ in range(50):
		q0 = torch.mul(mattrans[0],v)
		q0 = q0.sum(axis=-1)
		q1 = torch.mul(mattrans[1],v)
		q1 = q1.sum(axis=-1)
		q2 = torch.mul(mattrans[2],v)
		q2 = q2.sum(axis=-1)
		q3 = torch.mul(mattrans[3],v)
		q3 = q3.sum(axis=-1)
		q4 = torch.mul(mattrans[4],v)
		q4 = q4.sum(axis=-1)

		#print("q0 shape:",q0.shape)


		Q = torch.stack((q0,q1,q2,q3,q4),axis=-1)
		#print("Q:",Q.shape)
		pytorch_sm = nn.Softmax(dim=1)
		pi = pytorch_sm(beta*Q)
		#print("pi:",pi.shape)
		#print(pi[:][-1])

		next_q = (Q*pi).sum(axis=1)
		#print("next q:",next_q.shape)
		v = rffk + gamma * next_q
		#print("new v:",v.shape)
		#print("requires_grad maybe:",v.requires_grad)

	return(pi,Q)

def findpol(grid,pi,r,c):
  if grid[r][c] != 6: return
  maxprob = max(pi[r*ncols+c,:])
  a = 6
  for ana in range(5):
    if pi[r*ncols+c, ana] == maxprob: a = ana
  grid[r][c] = a
  r += acts[a][0]
  c += acts[a][1]
  findpol(grid,pi,r,c)

def plotpolicy(pi,startr=0,startc=0):
  grid = []
  for r in range(nrows):
    line = []
    for c in range(ncols):
      line += [cur_grid_map[r][c]+6]
    grid += [line]
  findpol(grid,pi,startr,startc)
  for r in range(nrows):
    line = ""
    for c in range(ncols):
      line += '^>v<x?01234'[grid[r][c]]
    print(line)

########### Learning reward function ################

from functools import reduce
 
#  0      1     2     3     4
# up, right, down, left, stay
act_map = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
act_name = ["UP","RIGHT","DOWN","LEFT","STAY"]
def proccess_act_input():
	k = input("Action: ")
	if k =="w":
		return 0
	elif k =="d":
		return 1
	elif k == "s":
		return 2
	elif k =="a":
		return 3
	elif k == "stop":
		return -1
	else:
		return 4

#TEST
'''
print("Starting a test")
cur_state = [0,0]
print(cur_state)
viz_map(cur_state[0],cur_state[1])
print("right")
print(cur_state)
cur_state = [a+b for a,b in zip(cur_state,act_map[1])]
viz_map(cur_state[0],cur_state[1])
print("Down")
print(cur_state)
cur_state = [a+b for a,b in zip(cur_state,act_map[2])]
viz_map(cur_state[0],cur_state[1])
print("Left")
print(cur_state)
cur_state = [a+b for a,b in zip(cur_state,act_map[3])]
viz_map(cur_state[0],cur_state[1])
print("Up")
print(cur_state)
cur_state = [a+b for a,b in zip(cur_state,act_map[0])]
viz_map(cur_state[0],cur_state[1])
print("stay")
print(cur_state)
cur_state = [a+b for a,b in zip(cur_state,act_map[4])]
viz_map(cur_state[0],cur_state[1])
'''


learning_rate = 0.001
 
#initial random guess on r
#r = np.random.rand(5)*2-1
r = np.zeros(5)
rk = torch.tensor(r,dtype=dtype,requires_grad=True)
 

tensor_time = []
for iter in range(10):
	#Collect Trajectories
	#start_state = [random.randint(0,nrows-1),random.randint(0,ncols-1)]
	viz_start = viz_starts[grid_idx]
	start_state = state_starts[grid_idx]
	cur_state = copy.copy(start_state)

	viz_map(cur_state[0],cur_state[1])


	act = proccess_act_input()
	trajacts = [act]
	print(act)
	cur_state = [a+b for a,b in zip(cur_state,act_map[act])]
	print(act_name[act])

	for step in range(15):
		viz_map(cur_state[0],cur_state[1])
		act = proccess_act_input()
		if act == -1:
			break
		trajacts.append(act)
		print(act)
		cur_state = [a+b for a,b in zip(cur_state,act_map[act])]
		print(act_name[act])

	trajcoords = reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [start_state])

	#LEARNING OCCURING
	print("LEARNING! ****************************************************************************************")
	for k in range(100):
		piout, Qout = forward(rk)
		
		loss = 0
		for i in range(len(trajacts)):
			acti = trajacts[i]
			state = trajcoords[i]
			loss += -torch.log(piout[state[0]*ncols+state[1]][acti])


		loss.backward()
		with torch.no_grad():
			grads_value = rk.grad
			rk -= learning_rate * grads_value
			rk.grad.zero_()

	#rk -= learning_rate * grads_value
	tensor_time.append(copy.deepcopy(rk))
	if iter % 1 == 0:
		#print(loss, rk)
		print(tensor_time)
		plotpolicy(piout,viz_start[0],viz_start[1])
		
		if input("switch maps?: ") == "yes":
			grid_idx += 1
			cur_grid_map = all_grids[grid_idx]
			viz_start = viz_starts[grid_idx]
		
