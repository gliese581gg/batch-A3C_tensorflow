'''
Worker module for Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
import numpy as np
import cv2
import tensorflow as tf
import time
import TF_utils_cjy as tu
import sys
import argparse
from preprocessing import preprocess
import env_way
import env_atari

class worker:
	def __init__(self,worker_idx,params,net,session,queue,worker_summary_dict,Control,target_net = None):
		self.dead = False
		self.params = params
		self.idx = worker_idx
		#environment
		if self.params['rom'] == 'toy_way':self.env = env_way.env_way(self.params)
		else : self.env=env_atari.env_atari(self.params)
		self.img = self.env.reset()

		#build networks
		self.net = net
		self.sess = session
		self.global_frame = net['global_frame']
		self.frame_ph = net['global_frame_ph']
		self.gf_op = net['global_frame_op']
		self.lr_ph = net['lr_ph']
		self.summary_op = worker_summary_dict['op']
		self.summary_writer = worker_summary_dict['writer']
		self.queue = queue
		self.control = Control
		
		if self.params['net_type'] == 'AnDQN' : 
			self.target = target_net
			eps_type = np.random.choice(np.arange(len(self.params['eps_prob'])),size=1,replace=True,p=np.array(self.params['eps_prob']))[-1]
			self.eps_max = self.params['eps_max'][eps_type]
			self.eps_min = self.params['eps_min'][eps_type]
			self.eps_frame = self.params['eps_frame'][eps_type]
		
		else : self.target = net #In A3C, the target network is local network (for code sharing with DQN)
		
		if self.idx == 0 and self.params['show_0th_thread'] : 
			cv2.startWindowThread()
			cv2.namedWindow('Worker'+str(self.idx)+'_screen')

	def run_worker(self,thread_idx):
		self.img = self.env.reset()
		epi_end = 0
		epi_reward = 0.
		num_epi = 0
		step = 0
		acc_step = 0
		gf = self.sess.run(self.net['global_frame'])
		start_time = time.time()
		acc_reward = 0. 
		if self.params['LSTM']:
			LSTM_h = np.zeros((1, self.params['dim_LSTM']))
			LSTM_c = np.zeros((1, self.params['dim_LSTM']))
			LSTM_h_temp = LSTM_h.copy()
			LSTM_c_temp = LSTM_c.copy()
		self.per = np.zeros((1,self.params['img_h'],self.params['img_w'],self.params['img_c']*self.params['history']))

		while gf < self.params['max_T']:

			if self.params['LSTM']:
				LSTM_h_temp = LSTM_h.copy()
				LSTM_c_temp = LSTM_c.copy()

			buffer_states = np.zeros((self.params['max_step'],self.params['img_h'],self.params['img_w'],self.params['img_c']*self.params['history']))	
			buffer_actions = np.zeros((self.params['max_step'],self.params['num_actions']))
			buffer_rewards = np.zeros((self.params['max_step'],1))
			buffer_used = np.zeros((self.params['max_step'],1))

			while step < self.params['max_step'] and epi_end == 0:

				if self.control[0] ==1 : continue

				self.per[0,:,:,0:self.params['img_c']*(self.params['history']-1)] = self.per[0,:,:,self.params['img_c']:self.params['img_c']*self.params['history']].copy()
				self.per[0,:,:,self.params['img_c']*(self.params['history']-1):] = preprocess(self.params,self.img)/255.0

				fd = {}
				fd[self.net['x']]=self.per
				fd[self.net['global_frame_ph']]=1

				if self.params['LSTM'] : 
					fd[self.net['LSTM_h_ph']] = LSTM_h_temp ; fd[self.net['LSTM_c_ph']] = LSTM_c_temp ; fd[self.net['unroll']] = np.array([1])
					pol,val,LSTM_c_h_temp,_inc_frame,gf = self.sess.run([self.net['policy'],self.net['value'],self.net['LSTM_state'],self.net['global_frame_op'],self.net['global_frame']],feed_dict = fd)

					LSTM_c_temp = LSTM_c_h_temp[0] 
					LSTM_h_temp = LSTM_c_h_temp[1] 
				else :
					pol,val,_inc_frame,gf = self.sess.run([self.net['policy'],self.net['value'],self.net['global_frame_op'],self.net['global_frame']],feed_dict = fd)

				pol=pol.reshape(-1);val=val.reshape(-1)	

				if acc_step % 200 == 0 and self.idx==0 : print '0th thread global_step/pol/val',gf,pol,val		
				
				if self.params['net_type'] == 'A3C':
					action = self.params['num_actions']-1
					seed = np.random.random()
					acc_prob = 0.

					for i in range(self.params['num_actions']):
						acc_prob += pol[i]
						if seed < acc_prob : action = i ; break

				elif self.params['net_type'] == 'AnDQN':
					eps = max(self.eps_min,self.eps_max - float(gf)/self.eps_frame)
					if np.random.uniform(0.0,1.0) <= eps: action = np.random.randint(0,self.params['num_actions'])
					else: 
						value = self.sess.run(self.net['value'],feed_dict = fd)
						action = np.argmax(val)

				step_reward = 0.
				epi_end = 0.

				if self.idx == 0 and self.params['show_0th_thread'] : 
					cv2.imshow('Worker'+str(self.idx)+'_screen',cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))

				self.img,step_reward,epi_end,info=self.env.step(action)
				
				real_step_reward = step_reward
				if self.params['clip_reward'] : 
					if step_reward > 1.0 : step_reward = 1.0
					if step_reward < -1.0 : step_reward = -1.0

				buffer_states[step] = self.per.copy()
				buffer_actions[step,action] = 1.
				buffer_rewards[step] = step_reward
				buffer_used[step] = 1
				step+=1 ; acc_step+=1 ; epi_reward += real_step_reward 

			Snext = self.per.copy()
			Snext[0,:,:,0:self.params['img_c']*(self.params['history']-1)] = Snext[0,:,:,self.params['img_c']:self.params['img_c']*self.params['history']].copy()
			Snext[0,:,:,self.params['img_c']*(self.params['history']-1):] = preprocess(self.params,self.img)/255.0

			if self.params['LSTM']:	#In A3C, target network is local network (Just for code sharing with DQN)
				Vnext = self.sess.run(self.target['value'],{self.target['x']:Snext,self.target['LSTM_h_ph']:LSTM_h_temp,self.target['LSTM_c_ph']:LSTM_c_temp,self.target['unroll']:np.array([1])})
			else : 
				Vnext = self.sess.run(self.target['value'],{self.target['x']:Snext})

			if self.params['net_type'] == 'AnDQN' : Vnext = np.max(Vnext) #TODO double dqn				

			R = (1-epi_end)*Vnext.reshape(-1)
			buffer_returns = buffer_rewards.copy()

			for i in range(step-1,-1,-1):
				buffer_returns[i] += self.params['discount']*R
				R = buffer_returns[i]
			if len(self.queue) < self.params['queue_max'] :
				if self.params['LSTM'] : 
					self.queue.append([buffer_states,
							buffer_actions,
							buffer_returns,
							buffer_used,
							step,LSTM_h,LSTM_c])
					LSTM_h = LSTM_h_temp.copy() ; LSTM_c = LSTM_c_temp.copy() 
				else : self.queue.append([buffer_states,
							buffer_actions,
							buffer_returns,
							buffer_used,
							step])


			if epi_end == 1:
				self.img = self.env.reset()
				self.per = np.zeros((1,self.params['img_h'],self.params['img_w'],self.params['img_c']*self.params['history']))	
				LSTM_h = np.zeros((1, self.params['dim_LSTM']))
				LSTM_c = np.zeros((1, self.params['dim_LSTM']))
				summary = tf.Summary()
				summary.value.add(tag='episode_reward', simple_value=float(epi_reward))
				self.summary_writer.add_summary(summary, gf)	
				self.summary_writer.flush()

				acc_reward += epi_reward
				epi_end = 0
				epi_reward = 0
				num_epi += 1


			if num_epi >= self.params['score_display_interval'] and self.idx == 0:
				print 'Total running time : ' + str(time.time()-start_time) + ' (# of frames : ' + str(gf) +' / ' + str(self.params['max_T'])+')'
				print '    0th thread average_reward : ' + str(acc_reward/max(1,num_epi)) + ' (' + str(num_epi) + ' episodes)'
				acc_reward = 0.
				num_epi = 0
			step = 0	

		

