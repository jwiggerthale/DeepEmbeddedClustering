
'''
Python implementation of linear sum assignment 
Required in modules 
If scipy is available use scipy implementation instead
'''
import numpy as np
import math 

class Solution:

	def __init__(self):
		
		# cost matrix
		self.cost = [[]]*31
		for i in range(31):
			self.cost[i] = [0]*31

		self.n = 0; # n workers and n jobs
		self.max_match = 0; # n workers and n jobs
		self.lx = [0]*31 # labels of X and Y parts
		self.ly = [0]*31 # labels of X and Y parts
		self.xy = [0]*31 # xy[x] - vertex that is matched with x,
		self.yx = [0]*31 # yx[y] - vertex that is matched with y
		self.S = [False]*31 # sets S and T in algorithm
		self.T = [False]*31 # sets S and T in algorithm
		self.slack = [0]*31 # as in the algorithm description
		self.slackx = [0]*31 # slackx[y] such a vertex, that
		self.prev_ious = [0]*31 # array for memorizing alternating p
		

	def init_labels(self):
		for i in range(len(self.lx)):
			self.lx[i] = 0
		for i in range(len(self.ly)):
			self.ly[i] = 0	

		
		for x in range(self.n):
			for y in range(self.n):
				self.lx[x] = max(self.lx[x], self.cost[x][y])
	
	
	def update_labels(self):
		x = 0
		y = 0
		delta = 99999999 # init delta as infinity
		
		for y in range(self.n): # calculate delta using slack
			if self.T[y] == False:
				delta = math.min(delta, self.slack[y])
				
				
		for x in range(self.n):
			if self.S[x] == True: # update X labels
				self.lx[x] -= delta

		for y in range(self.n):
			if self.T[y] == True:
				self.ly[y] += delta # update Y labels
		
		for y in range(self.n):
			if self.T[y] == False:
				self.slack[y] -= delta # update slack array
	
	
	def add_to_tree(self, x, prev_iousx):
		
	# x - current vertex,prev_iousx - vertex from X before x in the alternating path,
	# so we add edges (prev_iousx, xy[x]), (xy[x], x)
		self.S[x] = True # add x to S
		self.prev_ious[x] = prev_iousx # we need this when augmenting
		for y in range(self.n):
			if self.lx[x] + self.ly[y] - self.cost[x][y] < self.slack[y]:
				self.slack[y] = self.lx[x] + self.ly[y] - self.cost[x][y]
				self.slackx[y] = x
	
	
	
	def augment(self): # main function of the algorithm
		if self.max_match == self.n:
			return # check whether matching is already perfect
		x= 0
		y = 0
		root = 0 # just counters and root vertex
		q= [0]*31
		wr = 0
		rd = 0 # q - queue for bfs, wr,rd - write and read
		# pos in queue
		for i in range(len(self.S)):
			self.S[i] = False # init set S
		for i in range(len(self.T)):
			self.T[i] = False # init set T
		for i in range(len(self.prev_ious)):
			self.prev_ious[i] = -1 # init set S
		
		for x in range(self.n): # finding root. 
			if self.xy[x] == -1:
				q[wr] = root = x
				wr = wr + 1
				self.prev_ious[x] = -2
				self.S[x] = True
				break

		for y in range(self.n): #initializing slack array
			self.slack[y] = self.lx[root] + self.ly[y] - self.cost[root][y]
			self.slackx[y] = root
		
		# second part of augment() function
		while (True): # main cycle
		
			while (rd < wr): # building tree with bfs cycle
				x = q[rd]
				rd = rd + 1 # current vertex from X part
				for y in range(self.n): # iterate through all edges in equality graph
					if self.cost[x][y] == self.lx[x] + self.ly[y] and self.T[y] == False:
						if self.yx[y] == -1:
							break # an exposed vertex in Y found, so
												# augmenting path exists!
						self.T[y] = True # else just add y to T,
						q[wr] = self.yx[y] # add vertex yx[y], which is matched
						wr = wr + 1
						# with y, to the queue
						self.add_to_tree(self.yx[y], x) # add edges (x,y) and (y,yx[y]) to the tree
		
				if y < self.n:
					break # augmenting path found!
			if y < self.n:
				break # augmenting path found!
			
			self.update_labels() #augmenting path not found, so improve labeling
			
			wr = 0
			rd = 0
		
			for y in range(self.n):
				if self.T[y] == False and self.slack[y] == 0:
					if self.yx[y] == -1: # exposed vertex in Y found - augmenting path exists!
						x = self.slackx[y]
						break
					else:
						self.T[y] = True # else just add y to T,
						if self.S[self.yx[y]] == False:
					
							q[wr] = self.yx[y] # add vertex yx[y], which is matched with
							wr = wr + 1
							# y, to the queue
							self.add_to_tree(self.yx[y], self.slackx[y]); # and add edges (x,y) and (y,
							# //yx[y]) to the tree
			if y < self.n:
				break # augmenting path found!
		
		if y < self.n: # we found augmenting path!
			self.max_match = self.max_match + 1 # increment matching
			# in this cycle we inverse edges along augmenting path
			cx = x
			cy = y
			ty = 0
			while(cx != -2):
				
				ty = self.xy[cx]
				self.yx[cy] = cx
				self.xy[cx] = cy
			
				cx = self.prev_ious[cx]
				cy = ty
			self.augment() #recall function, go to step 1 of the algorithm
		# end of augment() function
	
	def hungarian(self):
		
		ret = 0 # weight of the optimal matching
		self.max_match = 0; # number of vertices in current matching
		for i in range(len(self.xy)):
			self.xy[i] = -1
		for i in range(len(self.yx)):
			self.yx[i] = -1

		self.init_labels() # step 0
		self.augment() # steps 1-3
		
		for x in range(self.n):
			ret += self.cost[x][self.xy[x]]
		
		return ret
	
	def assignmentProblem(self, Arr, N):
		
		self.n = N
		for i in range(self.n):
			for j in range(self.n):
				self.cost[i][j] = -1*Arr[i*self.n + j]
				
		ans = -1 * self.hungarian()
		
		return ans - 2000


# The code is contributed by Nidhi goel. 
