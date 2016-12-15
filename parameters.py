'''
Hyperparameters for Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
def load_params():
	params = {
		#Meta
		'log_name':'A3C', #the tensorboard event files will be saved in the folder 'params[log_path]+params[log_name]'
		'log_path':'logs/',
		'ckpt_path':'ckpt/', #the checkpoint files will be saved in params['ckpt_path']
		'ckpt_file' : None, #checkpoint file name (including full path)
		'eval_mode':False, #no training. only evaluations.
		'eval_wait':0., #seconds. waiting time between frames in evaluation mode.
		#environment
		'rom' : 'pong', #atari rom name (without '.bin') or your environment name. 
		'show_0th_thread' : False, #display 0th thread's screen
		'show_eval' : False, #display evaluation screen
		'frameskip' : 4,  #frameskip in atari
		'num_actions' : 0, #number of possible actions (automatically detected if the environment class has 'action_space.n')
		'img_w' : 84,
		'img_h' : 84,
		'img_c' : 1,
		'history' : 1, #number of frames aggregated into one network input
		'repeat_prob':0., #action repeat probability in atari
		#Networks
		'net_type' : 'A3C', #network type : A3C or AnDQN
		'num_workers': 16, #number of threads
		'convs_size': [3,3,3,3], #filter sizes in conv layers. len(params['convs_size']) is the number of convolutional layers
		'convs_filter' : [32,32,32,32], #output channel in conv layers
		'convs_stride' : [2,2,2,2], #stride in conv layers
		'dim_fc' : [], #dimensions of fc layers. len(params['dim_fc']) is the number of fc layers. it does not include LSTM layer.
		'dim_LSTM':256, #cell size of LSTM. used when params['LSTM'] is True
		'entropy_reg_coeff':0.01, #entropy regularization coefficient in A3C
		'max_step' : 20, #unroll in LSTM
		'LSTM' : True, #whether or not use the final LSTM layer
		'activations':'elu', #see TF_utils_cjy.py for possible activations.
		#training
		'queue_max' : 20, #maximum minibatch size
		'queue_min' : 0, #minimum minibatch size
		'discount' : 0.99,
		'lr' : 1e-4, #calculated using loguniform.
		#'lr_loguniform_low':1e-4, #low bound of log-uniform learning rate sampling
		#'lr_loguniform_high':1e-2, #upper bound of log-uniform learning rate sampling
		#'lr_loguniform_seed':0.1,#0.4226, #You can use random value between 0.0~1.0
		#'lr_zero_frame' : 30*4*(10**6), #learning rate will be annealed lineary and it will be zero after this frame.
		'rms_decay':0.99, #not used
		'rms_momentum':0.0, #not used
		'rms_eps':0.1, #not used
		'max_T' : 30*4*(10**6), #The learning ends in this frame.
		'score_display_interval' : 5, #episodes. display 0th thread's mean score after every 'score_display_interval' episodes
		'save_interval' : 1000000, #frames. save the check point in every 'save_interval' frames
		'eval_interval' : 30*4*(10**6)-2,#frames. Pause the learning and do the evaluation in every 'eval_interval' frames
		'eval_duration' : 20,#episodes. number of evaluation episodes.
		'clip_grad' : True, #whether or not clip the gradient
		'grad_clip_norm' : 40., #norm bound for gradient
		'clip_reward' : True, #clip reward to [-1.~1.]
		'eps_max' : [1.0,1.0,1.0], #maximum epsilon in AnDQN
		'eps_min' : [0.1,0.01,0.5], #minimum epsilon in AnDQN
		'eps_frame' : [1000000,1000000,1000000], #epsilon will be annealed lineary and it will be 'eps_min' after this frame.
		'eps_prob' : [0.4,0.3,0.3], #every thread pick one of epsilon settings with this distribution
		'target_copy_interval' : 30000, #interval between target net copy in AnDQN
		}

	if params['rom'] == 'toy_way':
		params['convs_size'] = []
		params['convs_filter'] = []
		params['convs_stride'] = []
		params['field_size'] = 4
		params['pixel_per_grid'] = 5
		params['history'] = 1
		params['num_waypoints'] = 1
		params['reward_move'] = -0.1
		params['reward_waypoint'] = 1.0
		params['reward_clear'] = 0.0
		params['reward_timeout'] = 0
		params['timeout'] = 100
		params['frameskip'] = 1
		params['img_h'] = params['field_size'] * params['pixel_per_grid']
		params['img_w'] = params['field_size'] * params['pixel_per_grid']
		params['lr_init'] = 0.0001
		params['eval_interval'] = 10000 #frames
		params['eval_duration'] = 10#episodes

	return params
