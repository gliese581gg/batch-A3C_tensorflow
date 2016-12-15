'''
Networks for Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
import numpy as np
import tensorflow as tf
import TF_utils_cjy as tu
from tensorflow.python.ops import rnn, rnn_cell

def build(params,net_name,device="/gpu:0"):
	activations = params['activations']
	print 'Building ' + net_name
	with tf.variable_scope(net_name) as vs:	
		#input
		x = tf.placeholder('float',[None,params['img_h'],params['img_w'],params['img_c']*params['history']],name='x') #batch*max_timestep,h,w,c
		action = tf.placeholder("float", [None, params['num_actions']],name='actions')
		returns = tf.placeholder("float",[None,1],name='returns')
		mask = tf.placeholder("float",[None,1],name='mask')
		batch_size = tf.placeholder("float",name='batch_size')
		unroll = tf.placeholder(tf.int32,[None])	

		#conv_layers
		with tf.variable_scope('fea') as vs_fea:
			convs = []
			convs_shapes = []
			inputs = x
			for i in range(len(params['convs_size'])):
				convs.append( tu.conv_layer('conv'+str(i),inputs,params['convs_filter'][i],params['convs_size'][i],params['convs_stride'][i],activation=activations) )
				convs_shapes.append( convs[-1].get_shape().as_list() )
				inputs = convs[-1]

			conv_flat,conv_flat_dim = tu.img_to_vec(inputs)		
	
		#common fc/lstm
		with tf.variable_scope('common') as vs_c:
			inputs = conv_flat
			fcs = []
			for i in range(len(params['dim_fc'])):
				fcs.append( tu.fc_layer('fc'+str(i),inputs,hiddens=params['dim_fc'][i],activation=activations)  )
				inputs = fcs[-1]

			if params['LSTM'] : 
				cells = rnn_cell.BasicLSTMCell(params['dim_LSTM'], forget_bias=1.0,state_is_tuple=True)
				LSTM_h_ph = tf.placeholder('float',[None,params['dim_LSTM']])  #batch,dim	
				LSTM_c_ph = tf.placeholder('float',[None,params['dim_LSTM']]) 	
				state_tuple = tf.nn.rnn_cell.LSTMStateTuple(LSTM_c_ph,LSTM_h_ph)
				inputs = tf.reshape(inputs,[tf.shape(unroll)[0],-1,inputs.get_shape().as_list()[-1]])
				LSTM_output, LSTM_state = tf.nn.dynamic_rnn(cells,inputs,initial_state = state_tuple,sequence_length = unroll)
				LSTM_output = tf.reshape(LSTM_output,[-1,params['dim_LSTM']])
				inputs = LSTM_output

		#State Value
		with tf.variable_scope('v') as vs_v:
			value = tu.fc_layer('value',inputs,hiddens=1,activation='linear')
		#Softmax_Policy
		with tf.variable_scope('p') as vs_p:
			policy = tu.fc_layer('policy',inputs,hiddens=params['num_actions'],activation='softmax')

		#Loss
		log_policy = tf.log(tf.clip_by_value(policy, 1e-4, 1.0))
		advantage = (returns - value)*mask 
		loss_ac_p = tf.reduce_sum(action*log_policy,1,keep_dims=True)*tf.stop_gradient(advantage)
		loss_ac_v = 0.25*tf.square(advantage)

		entropy = -tf.reduce_sum(log_policy*policy,1,keep_dims=True)
		loss_ac_p += params['entropy_reg_coeff']*entropy		

		loss_ac_p = -tf.reduce_sum(loss_ac_p) #Policy gradient uses gradient ascent (not descent!)
		loss_ac_v = tf.reduce_sum(loss_ac_v)

		loss_total = loss_ac_p+loss_ac_v
		#loss_total = loss_total/tf.reduce_sum(mask)
		#loss_total = loss_total/tf.to_float(tf.shape(unroll)[0])
	

	#grads
	vars_all = tu.get_all_var_from_net(net_name)	
	lr = tf.placeholder('float')
	#optimizer = tf.train.RMSPropOptimizer(lr,params['rms_decay'],params['rms_momentum'],params['rms_eps'],use_locking=False)
	optimizer = tf.train.AdamOptimizer(lr,use_locking=False)
	gvs = tf.gradients(loss_total,vars_all)
	if params['clip_grad']:		
		gvs,grad_global_norm = tf.clip_by_global_norm(gvs, params['grad_clip_norm'])

	gvs = list(zip(gvs, vars_all))
	train = optimizer.apply_gradients(gvs)
	global_frame = tf.Variable(0, name='global_step', trainable=False)
	frame_ph = tf.placeholder(tf.int32)
	with tf.device("/cpu:0"): 
		gf_op = tf.assign_add(global_frame,tf.reduce_sum(frame_ph))	


	output = {'x':x, 
		'action':action,
		'returns':returns,
		'policy':policy,
		'value':value,
		'loss_total':loss_total,
		'vars_all':vars_all,
		'grad' : gvs,
		'entropy':tf.reduce_sum(entropy)/tf.to_float(tf.shape(unroll)[0]),
		'loss_p' : loss_ac_p/tf.to_float(tf.shape(unroll)[0]),
		'loss_v' : loss_ac_v/tf.to_float(tf.shape(unroll)[0]),
		'train':train,
		'global_frame':global_frame,
		'global_frame_ph':frame_ph,
		'global_frame_op':gf_op,
		'lr_ph':lr,
		'mask_ph':mask,
		'batch_size':batch_size,
		'grad_norm':grad_global_norm,
		'unroll':unroll
		}

	if params['LSTM'] : output['LSTM_h_ph']= LSTM_h_ph ; output['LSTM_c_ph']= LSTM_c_ph ; output['LSTM_state']= LSTM_state

	return output
		
