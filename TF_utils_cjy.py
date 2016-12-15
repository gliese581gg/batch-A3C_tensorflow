'''
Tensorflow utils
Jinyoung Choi
'''
import tensorflow as tf
import numpy as np
import time

#All layers use xavier initialization
def conv_layer(layer_name,inputs,filters,size,stride,activation='lrelu'):
	channels = inputs.get_shape().as_list()[3]
	bound = 1.0/np.sqrt(size*size*channels)
	weight = tf.Variable(tf.random_uniform([size,size,int(channels),filters], minval=-bound,maxval=bound), name=layer_name+'/w')
	biases = tf.Variable(tf.random_uniform(shape=[filters], minval=-bound,maxval=bound), name=layer_name+'/b')

	conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='VALID', name=layer_name+'/c')	
	conv_biased = tf.add(conv,biases, name=layer_name+'/cb')	
	print '    Layer  "%s" : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d, activation = %s' % (layer_name,size,size,stride,filters,int(channels),activation)

	if activation == 'linear' : return conv_biased
	elif activation == 'relu' : return tf.nn.relu(conv_biased,name=layer_name+'/o')
	elif activation == 'lrelu' : return tf.maximum(0.1*conv_biased,conv_biased,name=layer_name+'/o')
	elif activation == 'elu' : return tf.nn.elu(conv_biased,name=layer_name+'/o')
	elif activation == 'tanh' : return tf.nn.tanh(conv_biased,name=layer_name+'/o')
	elif activation == 'sigmoid' : return tf.nn.sigmoid(conv_biased,name=layer_name+'/o')
	elif activation == 'softmax' : return tf.nn.softmax(conv_biased,name=layer_name+'/o')
	elif activation == 'softplus' : return tf.nn.softplus(conv_biased,name=layer_name+'/o')
	else : raise ValueError

def pooling_layer(layer_name,inputs,size,stride):
	print '    Layer  "%s" : Type = Pool, Size = %d * %d, Stride = %d' % (layer_name,size,size,stride)
	return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=layer_name+'/o')

def img_to_vec(inputs):
	input_shape = inputs.get_shape().as_list()
	dim = input_shape[1]*input_shape[2]*input_shape[3]
	return tf.reshape(inputs, [-1,dim]), dim

def fc_layer(layer_name,inputs,hiddens,activation='lrelu'):
	input_shape = inputs.get_shape().as_list()
	dim = input_shape[1]
	bound = 1.0/np.sqrt(dim)
	weight = tf.Variable(tf.random_uniform([dim,hiddens], minval=-bound,maxval=bound),name=layer_name+'/w')
	biases = tf.Variable(tf.random_uniform([hiddens], minval=-bound,maxval=bound),name=layer_name+'/b')	
	print '    Layer  "%s" : Type = Full, Hidden = %d, Input dimension = %d,  Activation = %s' % (layer_name,hiddens,int(dim),activation)	
	ip = tf.add(tf.matmul(inputs,weight),biases,name=layer_name+'/ip')
	if activation == 'linear' : return ip
	elif activation == 'relu' : return tf.nn.relu(ip,name=layer_name+'/o')
	elif activation == 'elu' : return tf.nn.elu(ip,name=layer_name+'/o')
	elif activation == 'lrelu' : return tf.maximum(0.1*ip,ip,name=layer_name+'/o')
	elif activation == 'tanh' : return tf.nn.tanh(ip,name=layer_name+'/o')
	elif activation == 'sigmoid' : return tf.nn.sigmoid(ip,name=layer_name+'/o')
	elif activation == 'softmax' : return tf.nn.softmax(ip,name=layer_name+'/o')
	elif activation == 'softplus' : return tf.nn.softplus(ip,name=layer_name+'/o')
	else : raise ValueError



def leaky_relu(layer_name,inputs,alpha):
	return tf.maximum(alpha*inputs,inputs,name=layer_name)

def get_all_var_from_net(net_name):
	with tf.variable_scope(net_name) as vs:
		return [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]

def get_all_var_from_layer(layer_name):
	with tf.variable_scope(layer_name) as vs:
		return [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]

def deconv_layer(layer_name,inputs,w_size,out_h,out_w,out_c,stride,activation='lrelu'):
	channels = inputs.get_shape().as_list()[3]
	bound = 2.0/np.sqrt(size*size*channels+size*size*out_c)
	weight = tf.Variable(tf.random_uniform([w_size,w_size,out_c,int(channels)], minval=-bound,maxval=bound), name=layer_name+'/w')
	biases = tf.Variable(tf.random_uniform([out_c], minval=-bound,maxval=bound), name=layer_name+'/b')

        deconv = tf.nn.conv2d_transpose(inputs, weight,[tf.shape(inputs)[0],out_h,out_w,out_c],strides=[1, stride, stride, 1], padding='SAME',name=layer_name+'/dc')
	deconv_biased = tf.add(deconv,biases, name=layer_name+'/db')	

	print '    Layer  "%s" : Type = Deconv, W_Size = %d * %d, Out_Size = %d * %d, Out_channels = %d, Stride = %d, activation = %s' % (layer_name,w_size,w_size,out_h,out_w,out_c,stride,activation)

	if activation == 'linear' : return deconv_biased
	elif activation == 'relu' : return tf.nn.relu(deconv_biased,name=layer_name+'/o')
	elif activation == 'lrelu' : return tf.maximum(0.1*deconv_biased,deconv_biased,name=layer_name+'/o')
	elif activation == 'elu' : return tf.nn.elu(deconv_biased,name=layer_name+'/o')
	elif activation == 'tanh' : return tf.nn.tanh(deconv_biased,name=layer_name+'/o')
	elif activation == 'sigmoid' : return tf.nn.sigmoid(deconv_biased,name=layer_name+'/o')
	elif activation == 'softmax' : return tf.nn.softmax(deconv_biased,name=layer_name+'/o')
	elif activation == 'softplus' : return tf.nn.softplus(deconv_biased,name=layer_name+'/o')
	else : raise ValueError

def display_num_params(net_name):
	_vars = get_all_var_from_net(net_name)
	num_par = 0
	for varidx in range(len(_vars)):
		ttt =  _vars[varidx].get_shape().as_list()
		print ttt
		tt = 0
		for asdf in range(len(ttt)):
			if tt == 0 : tt += ttt[asdf]
			else : tt *= ttt[asdf]
		num_par += tt
	print num_par



'''
#Test code...
x = tf.placeholder('float32',[None,84,84,4],name='x')
with tf.variable_scope('net1') as net:
	net1_conv1 = conv_layer('conv1',x,4,3,2)
	net1_pool2 = pooling_layer('pool2',net1_conv1,2,1)
	net1_fc3 = fc_layer('fc3',net1_pool2,30,True)

with tf.variable_scope('net2') as net:
	net2_conv1 = conv_layer('conv1',x,4,3,2)
	net2_pool2 = pooling_layer('pool2',net2_conv1,2,1)
	net2_fc3 = fc_layer('fc3',net2_pool2,30,True)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

v1 = get_all_var_from_net('net1')
v2 = get_all_var_from_net('net2')
v3 = get_all_var_from_layer('net1/conv1')
v4 = get_all_var_from_layer('net2/fc3')


for item in v1 : print item.name
for item in v2 : print item.name
for item in v3 : print item.name
for item in v4 : print item.name

'''






