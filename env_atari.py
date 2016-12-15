import numpy as np
import time
from ale_python_interface import ALEInterface

class c_action_space:
	def __init__(self,n):
		self.n=n

class env_atari:
	def __init__(self,params):
		self.params = params
		self.ale = ALEInterface()
		self.ale.setInt('random_seed',np.random.randint(0,500))
		self.ale.setFloat('repeat_action_probability',params['repeat_prob'])
		self.ale.setInt(b'frame_skip',params['frameskip'])
		self.ale.setBool('color_averaging', True)
		self.ale.loadROM('roms/'+params['rom']+'.bin')
		self.actions = self.ale.getMinimalActionSet()
		self.action_space = c_action_space(len(self.actions))
		self.screen_width,self.screen_height = self.ale.getScreenDims()

	def reset(self):
		self.ale.reset_game()
		seed = np.random.randint(0,7)
		for i in range(seed) : self.ale.act(0)
		return self.get_image()
		

	def step(self,action): 
		reward = self.ale.act(self.actions[action])
		next_s = self.get_image()
		terminate = self.ale.game_over()
		return next_s, reward, float(terminate),0

	def get_image(self):
		temp = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(temp)
		#self.ale.getScreenGrayscale(temp)
		return temp.reshape((self.screen_height, self.screen_width, 3))
		

