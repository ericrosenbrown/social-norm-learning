import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras.backend as K
 
# the empty grid
nrows = 5
ncols = 10
 
# ncats is the number of state categories
ncats = 5
 
# map state categories to states
# want m s.t. r %*% m = reward function
# first, just a map of the indexes
grid_map = np.array([ [0,0,0,0,0,2,0,0,1,0],
                 [0,1,0,0,0,2,0,0,0,0],
                 [0,0,0,0,0,2,1,3,0,0],
                 [0,0,0,1,0,2,0,3,0,0],
                 [0,0,0,0,0,2,0,3,0,4]])

 
colors = ["white", "blue", "orange", "yellow", "green"]
 
sns.heatmap(grid_map, cmap=sns.xkcd_palette(colors), yticklabels=False, xticklabels=False,
            annot=False, cbar = False, annot_kws={"size": 30}, linewidths=1, linecolor="gray")
plt.show()



# r is the rewards for the different location categories
r = np.array([0, -1, -1, -1, 10])
 
matmap = np.zeros((nrows,ncols,ncats))
for i in range(nrows):
  for j in range(ncols):
    for k in range(ncats):
      matmap[i,j,k] = 0+(grid_map[i,j] == k)
def clip(v,min,max):
  if v < min: v = min
  if v > max-1: v = max-1
  return(v)
 
acts = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
nacts = len(acts)
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


rk = K.placeholder(len(r))
rfk = K.dot(K.constant(matmap),K.reshape(rk,(-1,1)))
rffk = K.reshape(rfk,(-1,1))
 
v = K.reshape(rfk,(-1,1))
gamma = 0.90
beta = 10.0

for _ in range(50):
  q0 = K.dot(K.constant(mattrans[0]),v)
  q1 = K.dot(K.constant(mattrans[1]),v)
  q2 = K.dot(K.constant(mattrans[2]),v)
  q3 = K.dot(K.constant(mattrans[3]),v)
  q4 = K.dot(K.constant(mattrans[4]),v)
  Q = K.concatenate([q0,q1,q2,q3,q4])
  pi = K.softmax(beta*Q)
  v = rffk + gamma * K.reshape(K.sum(Q * pi,axis=1),(-1,1))


planner = K.function([rk], [pi, Q])

r = np.array([0, -1, -1, -1, 10])
piout, Qout = planner([r])



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

def plotpolicy(pi):
  grid = []
  for r in range(nrows):
    line = []
    for c in range(ncols):
      line += [6]
    grid += [line]
  findpol(grid,pi,0,0)
  for r in range(nrows):
    line = ""
    for c in range(ncols):
      line += '^>v<x? '[grid[r][c]]
    print(line)

r = np.array([0, -1, -1, -1, 10])
piout, Qout = planner([r])
plotpolicy(piout)



from functools import reduce
 
#  0      1     2     3     4
# up, right, down, left, stay
 
trajacts = [1,1,2,2,2,2,1,1,1,1,1,1,1,4,4,4,4,4,4,4]
 
trajcoords = reduce((lambda seq, a: seq+[[seq[len(seq)-1][0] + acts[a][0], seq[len(seq)-1][1] + acts[a][1]]]), trajacts, [[0,0]])

print(trajcoords)

loss = 0
for i in range(len(trajacts)):
  acti = trajacts[i]
  state = trajcoords[i]
  loss += -K.log(pi[state[0]*ncols+state[1]][acti])



learning_rate = 0.001
 
r = np.random.rand(5)*2-1
r = np.ones(5)

 
grads = K.gradients(loss, rk)[0]
iterate = K.function([rk], [loss, grads])
 
for iter in range(5000):
  loss_value, grads_value = iterate([r])
  r -= learning_rate * grads_value
  if iter % 100 == 0: print(loss_value, r)

