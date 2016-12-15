'''
Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
import numpy as np
import cv2
import tensorflow as tf
import time
import TF_utils_cjy as tu
import sys
import argparse
import env_way
import thread
import worker
import parameters
import env_atari
from preprocessing import preprocess

params = parameters.load_params()

ap = argparse.ArgumentParser()
ap.add_argument("-log", "--log_name", required = False, help = "log file name")
ap.add_argument("-net", "--net_type", required = False, help = "network type('A3C' or 'AnDQN')")
ap.add_argument("-LSTM", "--LSTM", required = False, help = "LSTM (True or False)")
ap.add_argument("-show_eval", "--show_eval", required = False, help = "show evaluation screen? (True or False)")
ap.add_argument("-eval_mode", "--eval_mode", required = False, help = "Evaluation only (True or False)")
ap.add_argument("-ckpt", "--ckpt_file", required = False, help = "checkpoint name (without path)")
ap.add_argument("-rom", "--rom", required = False, help = "game rom name without '.bin' ('toy_way' for toy problem)")
args = vars(ap.parse_args())
print args
for i in args.keys():
	if i in params.keys() and args[i] is not None:
		if args[i] == 'True' : aar = True
		elif args[i] == 'False' : aar = False
		else : aar = args[i]
		params[i] = aar

if params['eval_mode'] : params['num_workers'] = 0
#if params['LSTM'] : params['history'] = 1


#environment
if params['rom'] == 'toy_way':	env = env_way.env_way(params)
else : 
	env = env_atari.env_atari(params)
	#log-uniform learning rate setting (reference : https://github.com/miyosuda/async_deep_reinforce )
	#params['lr_init'] = np.exp(np.log(params['lr_loguniform_low']) * (1-params['lr_loguniform_seed']) + np.log(params['lr_loguniform_high']) * params['lr_loguniform_seed'])
img = env.reset()
params['num_actions'] = env.action_space.n


if params['show_eval'] : 
	cv2.startWindowThread()
	cv2.namedWindow('Evaluation')

#build_networks
if params['net_type'] == 'A3C':
	import Net_A3C
	net = Net_A3C.build(params,'A3C')
elif params['net_type'] == 'AnDQN':
	import Net_AnDQN
	net = Net_AnDQN.build(params,'AnDQN')
else : raise ValueError
workers = []

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

saver = tf.train.Saver() #TODO save only the master net
sess = tf.Session(config=gpu_config)
sess.run(tf.initialize_all_variables())

summary_op = tf.merge_all_summaries()
log = params['log_path']+params['log_name']
summary_writer = tf.train.SummaryWriter(log, sess.graph_def)

worker_summary_dict = {'op':summary_op,'writer':summary_writer}

if params['ckpt_file'] is not None : 
	print 'Continue from ',params['ckpt_file']
	saver.restore(sess,params['ckpt_file'])

#Create workers and Experience Queue
Exp_Queue = []
Control = [0] #shared across all threads. 0 means 'do training', 1 means 'pause'
for i in range(params['num_workers']):
		print 'Initializing Thread ' + str(i)
		# worker_idx,params,worker_net,copy_master_to_worker,copy_master_to_target,train_p,train_v,session,master,target,global_step
		workers.append(worker.worker(i,params,net,sess,Exp_Queue,worker_summary_dict,Control))
		thread.start_new_thread(workers[i].run_worker,(i,))

#Start training
gf = sess.run(net['global_frame'])
last_eval_frame = gf
last_save = gf
last_target_copy = gf

print 'Start learning. Hyper paramters are:'
print params

def do_training():
	global sess
	global net
	global Exp_Queue
	global params
	global gf
	queue_zip = zip(*Exp_Queue)
	states = list(queue_zip[0])
	actions = list(queue_zip[1])
	returns = list(queue_zip[2])
	mask = list(queue_zip[3])
	steps = list(queue_zip[4])
	batch_size = len(steps)
	if params['LSTM'] :
		lstm_hs = list(queue_zip[5])
		lstm_cs = list(queue_zip[6])
	del Exp_Queue[:]
	#if epi_end == 1 : print buffer_returns
	#lr = max(0.,params['lr_init']*float((params['lr_zero_frame']-gf))/float(params['lr_zero_frame']))
	lr = params['lr']

	states = np.concatenate(states,0)
	actions = np.concatenate(actions,0)
	returns = np.concatenate(returns,0)
	mask = np.concatenate(mask,0)
	steps = np.array(steps)
	if params['LSTM'] :
		lstm_hs = np.concatenate(lstm_hs,0)
		lstm_cs = np.concatenate(lstm_cs,0)

	tfd = {net['x'] : states,
		net['action'] : actions,
		net['returns'] : returns,
		net['lr_ph'] : lr,
		net['mask_ph'] : mask,
		net['batch_size']:batch_size,
		net['unroll'] : steps
		}
	if params['LSTM']:
		tfd[net['LSTM_h_ph']] = lstm_hs
		tfd[net['LSTM_c_ph']] = lstm_cs



	#_,e,gf,loss_v,loss_p,entropy = sess.run([net['train'],net['loss_total'],net['global_frame'],net['loss_v'],net['loss_p'],net['entropy']],tfd)
	_,e,gf,loss_v,loss_p,entropy,grad = sess.run([net['train'],net['loss_total'],net['global_frame'],net['loss_v'],net['loss_p'],net['entropy'],net['grad_norm']],tfd)

	summary = tf.Summary()
	summary.value.add(tag='loss_v', simple_value=float(loss_v))
	summary.value.add(tag='loss_p', simple_value=float(loss_p))
	summary.value.add(tag='entropy', simple_value=float(entropy))
	summary.value.add(tag='lr', simple_value=float(lr))
	summary.value.add(tag='minibatch_size', simple_value=float(steps.shape[0]))						
	summary.value.add(tag='grad_norm', simple_value=float(grad))

	summary_writer.add_summary(summary, gf)	
	summary_writer.flush()
	
	if e > 500 : raise ValueError



while gf < params['max_T'] :

	for ii in range(len(workers)):
		if workers[ii].dead : raise ValueError

	gf = sess.run(net['global_frame'])

	if len(Exp_Queue) > params['queue_min'] and Control[0] == 0  : do_training()

	if params['net_type'] == 'AnDQN':
		if gf > last_target_copy + params['target_copy_interval'] : 
			sess.run(net['copy_target'])
			last_target_copy = gf

	if gf > last_save + params['save_interval'] : 
		saver.save(sess, params['ckpt_path']+'ckpt_'+str(gf))
		print 'Model saved as ckpt/ckpt'+str(gf)
		last_save = gf

	if gf > last_eval_frame + params['eval_interval'] or params['eval_mode']:
		print 'Start Evaluation! (Training is stopped)'
		Control[0] = 1
		epi_reward = 0.
		acc_reward=0.
		num_epi = 0
		if params['LSTM'] : LSTM_h = np.zeros((1, params['dim_LSTM'])) ; LSTM_c = np.zeros((1, params['dim_LSTM']))
		img = env.reset()
		per = np.zeros((1,params['img_h'],params['img_w'],params['img_c']*params['history']))	
		eval_start_time = time.time()
		epi_end = 0
		while num_epi < params['eval_duration']:
			per[0,:,:,0:params['img_c']*(params['history']-1)] = per[0,:,:,params['img_c']:params['img_c']*params['history']].copy()
			per[0,:,:,params['img_c']*(params['history']-1):] = preprocess(params,img)/255.0

			fd = {}
			fd[net['x']]=per

			if params['LSTM'] : 
				fd[net['LSTM_h_ph']] = LSTM_h ; fd[net['LSTM_c_ph']] = LSTM_c ; fd[net['unroll']] = np.array([1])
				pol,val,LSTM_c_h_temp = sess.run([net['policy'],net['value'],net['LSTM_state']],feed_dict = fd)
				LSTM_c = LSTM_c_h_temp[0] 
				LSTM_h = LSTM_c_h_temp[1] 
			else :
				pol,val = sess.run([net['policy'],net['value']],feed_dict = fd)
			pol=pol.reshape(-1);val=val.reshape(-1)
	
			if params['net_type'] == 'A3C':
				action = params['num_actions']-1
				seed = np.random.random()
				acc_prob = 0.

				for i in range(params['num_actions']):
					acc_prob += pol[i]
					if seed < acc_prob : action = i ; break

			elif params['net_type'] == 'AnDQN':				
				action = np.argmax(val)

			step_reward = 0
			epi_end = 0

			if params['show_eval'] : cv2.imshow('Evaluation',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

			img,step_reward,epi_end,info=env.step(action)
				
			epi_reward += step_reward

			if epi_end == 1:
				img = env.reset()
				per = np.zeros((1,params['img_h'],params['img_w'],params['img_c']*params['history']))	
				LSTM_h = np.zeros((1, params['dim_LSTM']))
				LSTM_c = np.zeros((1, params['dim_LSTM']))

				print '    eval_episode_'+str(num_epi)+' score : ' + str(epi_reward)

				acc_reward += epi_reward
				epi_end = 0
				epi_reward = 0
				num_epi += 1

			time.sleep(params['eval_wait'])


		print 'Evaluation Running Time : ' + str(time.time()-eval_start_time) + ' (# of frames learned: ' + str(gf) +' / ' + str(params['max_T'])+')'
		print '    average_reward : ' + str(acc_reward/max(1,num_epi)) + ' (' + str(num_epi) + ' episodes)'
		print 'Continue learning!'
		summary_data = tf.Summary()
		summary_data.value.add(tag='Evaluation_mean_score', simple_value=float(acc_reward/max(1,num_epi)))
		summary_writer.add_summary(summary_data, gf)
		summary_writer.flush()
		last_eval_frame = gf
		Control[0] = 0






