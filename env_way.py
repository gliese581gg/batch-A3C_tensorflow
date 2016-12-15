import numpy as np
import time

class c_action_space:
	def __init__(self,n):
		self.n=n

class env_way:
	def __init__(self,params):
		self.p = params
		temp = np.arange(params['field_size'])
		self.poss = np.concatenate((np.repeat(temp,params['field_size']).reshape((-1,1)),np.tile(temp,params['field_size']).reshape((-1,1))),axis=1)
		self.action_space = c_action_space(4)
		self.reset()

	def reset(self):
		self.field = np.zeros((self.p['field_size'],self.p['field_size'],3))
		self.img = np.zeros((self.p['field_size']*self.p['pixel_per_grid'],self.p['field_size']*self.p['pixel_per_grid'],3))
		self.deploy()
		self.render()
		self.term = 0
		return self.percieve()
		

	def deploy(self):
		self.timestep = 0
		self.info_agent = np.zeros(2) #x,y
		self.info_objects = np.zeros((self.p['num_waypoints'],4)) #idx,x,y,clear
		self.info_objects[:,0] = np.arange(self.p['num_waypoints'])

		pos_permu = np.random.permutation(self.poss.shape[0])
		self.info_agent = self.poss[pos_permu[0]]
		self.info_objects[:,1:3] = self.poss[pos_permu[1:self.p['num_waypoints']+1]]
		return None

	def render(self): #agent : green, waypoints : red(25*idx)
		self.field = np.zeros((self.p['field_size'],self.p['field_size'],3))
		self.img = np.zeros((self.p['field_size']*self.p['pixel_per_grid'],self.p['field_size']*self.p['pixel_per_grid'],3))
		for i in range(self.p['num_waypoints']):
			if self.info_objects[i,3] == 1 : continue
			self.field[self.info_objects[i,1],self.info_objects[i,2],2] = 225.0/self.p['num_waypoints']*(i+1)
		self.field[self.info_agent[0],self.info_agent[1],1] = 255.0
		self.field[self.info_agent[0],self.info_agent[1],2] = 0.0
		for i in range(self.p['field_size']):
			for j in range(self.p['field_size']):
				self.img [i*self.p['pixel_per_grid']:(i+1)*self.p['pixel_per_grid'],j*self.p['pixel_per_grid']:(j+1)*self.p['pixel_per_grid'],:] = self.field[i,j,:]

	def step(self,action): #0 :up, 1 :down, 2:left, 3:right
		self.movereward = 0.0
		self.move(action)
		self.timestep += 1
		reward = self.check()
		self.render()
		#print reward, self.term
		return self.percieve(), reward, self.term,0

	def move(self,direction):
		x = self.info_agent[0]
		y = self.info_agent[1]

		x_shift = 0
		y_shift = 0
		if direction == 0 : x_shift = -1
		elif direction == 1 : x_shift = 1
		elif direction == 2 : y_shift = -1
		elif direction == 3 : y_shift = 1

		xd = x+x_shift
		yd = y+y_shift
		self.movereward += self.p['reward_move']
		if xd < 0 or xd >= self.p['field_size'] or yd < 0 or yd >= self.p['field_size'] : return None

		self.info_agent = np.array([xd,yd])


	def check(self):
		step_reward = 0.0
		for i in range(self.p['num_waypoints']):
			if self.info_objects[i,1] == self.info_agent[0] and self.info_objects[i,2] == self.info_agent[1]:
				if i == 0 or (self.info_objects[0:i,3] == 1).all() :
					self.info_objects[i,1:3] = -1
					self.info_objects[i,3] = 1
					step_reward += self.p['reward_waypoint']
					if i == self.p['num_waypoints']-1 : 
						step_reward += self.p['reward_clear']
						self.term = 1

		step_reward += self.movereward
		if self.timestep > self.p['timeout'] : 
			step_reward += self.p['reward_timeout']
			self.term = 1

		return step_reward
		

	def percieve(self):
		self.render()		
		return self.img.astype('uint8')
		

