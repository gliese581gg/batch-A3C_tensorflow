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

		with tf.variable_scope('main_net') as vs_main:

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
					state_tuple = tf.nn.rnn_cell.LSTMStateTuple(LSTM_h_ph,LSTM_c_ph)	
					LSTM = tf.reshape(inputs,[-1,params['max_step'],inputs.get_shape().as_list()[-1]])
					unroll = tf.placeholder(tf.int32,[None])
					LSTM_output, LSTM_state = tf.nn.dynamic_rnn(cells,inputs,initial_state = state_tuple,sequence_length = unroll)
					LSTM_output = tf.reshape(LSTM_output,[-1,params['dim_LSTM']])
					inputs = LSTM_output

			#State Action Value
			with tf.variable_scope('v') as vs_v:
				value = tu.fc_layer('value',inputs,hiddens=params['num_actions'],activation='linear')

		with tf.variable_scope('target_net') as vs_target:

			#conv_layers
			with tf.variable_scope('fea') as vs_fea:
				target_convs = []
				target_convs_shapes = []
				inputs = x
				for i in range(len(params['convs_size'])):
					target_convs.append( tu.conv_layer('conv'+str(i),inputs,params['convs_filter'][i],params['convs_size'][i],params['convs_stride'][i],activation=activations) )
					target_convs_shapes.append( target_convs[-1].get_shape().as_list() )
					inputs = target_convs[-1]

				target_conv_flat,target_conv_flat_dim = tu.img_to_vec(inputs)		
	
			#common fc/lstm
			with tf.variable_scope('common') as vs_c:
				inputs = target_conv_flat
				target_fcs = []
				for i in range(len(params['dim_fc'])):
					target_fcs.append( tu.fc_layer('fc'+str(i),inputs,hiddens=params['dim_fc'][i],activation=activations)  )
					inputs = target_fcs[-1]


				if params['LSTM'] : 
					target_cells = rnn_cell.BasicLSTMCell(params['dim_LSTM'], forget_bias=1.0,state_is_tuple=True)
					target_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(LSTM_h_ph,LSTM_c_ph)	
					target_LSTM = tf.reshape(inputs,[-1,params['max_step'],inputs.get_shape().as_list()[-1]])
					target_LSTM_output, target_LSTM_state = tf.nn.dynamic_rnn(target_cells,inputs,initial_state = target_state_tuple,sequence_length = unroll)
					target_LSTM_output = tf.reshape(target_LSTM_output,[-1,params['dim_LSTM']])
					inputs = target_LSTM_output

			#State Action Value
			with tf.variable_scope('v') as vs_v:
				target_value = tu.fc_layer('value',inputs,hiddens=params['num_actions'],activation='linear')
			
		#Loss
		diff_q = (returns - value)*mask 
		loss_ac_p = tf.reduce_sum(action*log_policy,1,keep_dims=True)*tf.stop_gradient(advantage)
		loss_ac_v = 0.25*tf.square(advantage)

		entropy = -tf.reduce_sum(log_policy*policy,1,keep_dims=True)
		loss_ac_p += params['entropy_reg_coeff']*entropy		

		loss_ac_p = -tf.reduce_sum(loss_ac_p) #Policy gradient uses gradient ascent (not descent!)
		loss_ac_v = tf.reduce_sum(loss_ac_v)

		loss_total = loss_ac_p+loss_ac_v
		loss_total = loss_total/tf.reduce_sum(mask)
	

	#grads
	vars_all = tu.get_all_var_from_net(net_name)	
	lr = tf.placeholder('float')
	#optimizer = tf.train.RMSPropOptimizer(lr,params['rms_decay'],params['rms_momentum'],params['rms_eps'],use_locking=False)
	optimizer = tf.train.AdamOptimizer(lr,use_locking=False)
	gvs = tf.gradients(loss_total,vars_all)
	if params['clip_grad']:		
		#gvs = [(tf.clip_by_norm(grad, params['grad_clip_norm']), var) for grad, var in gvs]
		gvs = tf.clip_by_global_norm(gvs, params['grad_clip_norm'])[0]

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
		'entropy':tf.reduce_sum(entropy)/tf.reduce_sum(mask),
		'loss_p' : loss_ac_p/tf.reduce_sum(mask),
		'loss_v' : loss_ac_v/tf.reduce_sum(mask),
		'train':train,
		'global_frame':global_frame,
		'global_frame_ph':frame_ph,
		'global_frame_op':gf_op,
		'lr_ph':lr,
		'mask_ph':mask,
		'batch_size':batch_size
		}

	if params['LSTM'] : output['unroll'] = unroll; output['LSTM_h_ph']= LSTM_h_ph ; output['LSTM_c_ph']= LSTM_c_ph ; output['LSTM_state']= fc2_state

	return output
		
