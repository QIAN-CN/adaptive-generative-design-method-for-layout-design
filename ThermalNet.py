#ThermalNet.py
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
tf.reset_default_graph() 
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


me0=0.00
std0=0.10

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return(tf.Variable(initial))

def bias_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05) 
	return(tf.Variable(initial))
	
def conv2d(x, W, s=[1,1,1,1], padding='SAME'):
	if (padding.upper() == 'VALID'):
		return (tf.nn.conv2d(x,W,strides=s,padding='VALID'))
	# SAME
	return (tf.nn.conv2d(x,W,strides=s,padding='SAME'))
	
def Average_Pooling(x,in_cn):
	return tf.reshape(tf.nn.avg_pool(x,[1,10,10,1],[1,10,10,1],'SAME'),[-1, in_cn])

def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.nn.tanh(x)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def bn(x):
	return tf.layers.batch_normalization(x,training=False)

def dropout(x, keep_prob=0.5):
	return tf.nn.dropout(x, keep_prob)

def fc(x, in_cn, out_cn, name_scope, drop_out=False):
	with tf.variable_scope(name_scope):
		w = weight_variable([in_cn, out_cn])
		b = bias_variable([out_cn])
		h = relu(tf.matmul(x, w) + b)
	if drop_out:
		return dropout(h)
	else:
		return h

def seblock(x, in_cn):

	squeeze = Average_Pooling(x,in_cn)  
	with tf.variable_scope('sq'):
		w = weight_variable([in_cn, in_cn//16])
		b = bias_variable([in_cn//16])
		h = tf.matmul(squeeze, w) + b  
		excitation = relu(h)  
 
	with tf.variable_scope('ex'):
		w = weight_variable([in_cn//16, in_cn])  
		b = bias_variable([in_cn])
		h = tf.matmul(excitation, w) + b
		excitation = sigmoid(h)  
		excitation = tf.reshape(excitation, [-1, 1, 1, in_cn])  

	return x * excitation


def residual_block(x, cn, scope_name):
	with tf.variable_scope(scope_name):
		shortcut = x 
		x1 = bn(my_conv2d(x, cn,cn,me0,std0,3,3,1,1,'SAME',tf.nn.relu)) 
		x2 = bn(my_conv2d(x1,cn,cn,me0,std0,3,3,1,1,'SAME',None)) 
		x3 = seblock(x2, cn) # None*6*8*128

	return x3 + shortcut


def my_conv2d(x,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,pad='SAME',activation_L='None'):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,in_ch,out_ch],mean = me,stddev = std))
    b=tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L is None:
        L = tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b
    else:  
        L = activation_L(tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b)
    return L

def my_conv2d_transpose(x,out_wid,out_hei,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,activation_L1='None'):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,out_ch,in_ch],mean = me,stddev = std))
    output_shape=[tf.shape(x)[0],out_wid,out_hei,out_ch]
    #print(output_shape)
    b=tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L1 is None:
        L = tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b
    else:
        L =tf.add(activation_L1(tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b),tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME"))
    return L

def model(x):
  #x  None*10*10*1
	x1 = my_conv2d(x, 1,8,me0,std0,4,4,1,1,"SAME",tf.nn.tanh)   #None*10*10*8
	x2 = my_conv2d(x1, 8,16,me0,std0,3,3,1,1,"SAME",tf.nn.tanh) # None*10*10*16
	x3 = my_conv2d(x2, 16,32,me0,std0,3,3,1,1,'SAME',tf.nn.tanh) # None*10*10*32

	x4 = residual_block(x3, 32, 'res1') # None*10*10*32
	x5 = residual_block(x4, 32, 'res2') # None*10*10*32
	x6 = residual_block(x5, 32, 'res3') # None*10*10*32
 
	x7 =  my_conv2d_transpose(x6,50,50,32,32, me0,std0,5,5,5,5,tf.nn.tanh) # None*100*100*32 
	x8 =  my_conv2d_transpose(x7,100,100,32,16, me0,std0,3,3,2,2,tf.nn.tanh) # None*100*100*32
	x9 =  my_conv2d_transpose(x8,200,200,16,8, me0,std0,3,3,2,2,tf.nn.tanh) # None*200*200*16	
	x10 = my_conv2d(x9,8,1,me0,std0,3,3,1,1,'SAME',tf.nn.tanh)#tanh(conv2d(x12, w5)) # None*200*200*1
	print('x10shape=',x10.shape)
	return x10, x3, x2, x1


def first(argv = None):

	f10_name = 'dens_all.npy'
	f20_name = 'T_field_all.npy'

	batch_size = 10 #256
	decay_steps = 2000
	decay_rate = 0.99
	starter_learning_rate = 1e-3

	n_epochs = 50 #

	#define the size of graph
	nelx=10
	nely=10
	resolution_in = nelx*nely
    
	height = 200
	width =200
	resolution_out = height*width
	dens = np.load(f10_name)#

	Temp = np.load(f20_name)#

	data_size=np.size(dens,0)

	test_size=50 #
	train_size=250   #

	dens_train0=dens[0:train_size,:]-0.5
	Temp_train0=2*(Temp[0:train_size,:]-298)/(350-298)-1#
 
	dens_test0=dens[data_size-2*test_size:data_size-test_size,:]-0.5
	Temp_test0=2*(Temp[data_size-2*test_size:data_size-test_size,:]-298)/(350-298)-1#

	del dens
	del Temp

	xs = tf.placeholder(tf.float32, shape=[None, resolution_in],name='xs_node')
	xs_reshape = tf.reshape(xs, shape=[-1, nely, nelx, 1])
	ys = tf.placeholder(tf.float32, shape=[None, resolution_out])
	prediction_matrix, x3, x2, x1 = model(xs_reshape)
	prediction = tf.reshape(prediction_matrix, shape=[-1,resolution_out])

    
	mse = tf.losses.mean_squared_error(ys,prediction)
	mae = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction)))
	#loss
	loss = mse

	#train rate    
	global_step = tf.Variable(0, trainable=False)
	add_global = global_step.assign_add(1)
	learning_rate = tf.train.exponential_decay(starter_learning_rate,
		global_step=global_step,
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		staircase=False)
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	
	cgt_var=tf.global_variables()
	saver = tf.train.Saver([var for var in cgt_var if  var in tf.get_collection_ref("p_var") or var.name.startswith("res")])
    
	start_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
	#session
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		if True:
			saver.restore(sess,'./savemodel/model')
		batch_test = 50 #256
		mse_train = np.zeros(n_epochs)
		mse_test = np.zeros(n_epochs)
		mae_train = np.zeros(n_epochs)
		mae_test = np.zeros(n_epochs)
		prediction_num = 50 # < batch_test	
		iter_num = train_size // batch_size
		test_iter_num = test_size//batch_test
		order = np.arange(train_size)
		print('Training...')
		for epoch in range(n_epochs):
			total_mse = 0
			total_mae = 0
			test_total_mse = 0
			test_total_mae = 0
			np.random.shuffle(order)
			dens_train = dens_train0[order,:]
			Temp_train = Temp_train0[order,:]
			for iter_train in range(iter_num):
				x_batch = dens_train[iter_train*batch_size:(iter_train+1)*batch_size,:] #
				y_batch = Temp_train[iter_train*batch_size:(iter_train+1)*batch_size,:] #
				_, l_rate= sess.run([add_global, learning_rate,], feed_dict={xs: x_batch,ys:y_batch})
				_, batch_loss, batch_mae= sess.run([train_step, mse, mae], feed_dict={xs:x_batch,ys:y_batch})
				total_mse += batch_loss
				total_mae += batch_mae
			print('Epoch:',epoch,', Learning rate:',l_rate)

			mse_train[epoch] = total_mse/iter_num
			mae_train[epoch] = total_mae/iter_num
			print('MSE_train:', mse_train[epoch], end = ' ')
			print('MAE_train:', mae_train[epoch])


			for iter_test in range(test_iter_num):
				x_test = dens_test0[iter_test*batch_test:iter_test*batch_test+batch_test,:] #np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,0:resolution_in]
				y_test = Temp_test0[iter_test*batch_test:iter_test*batch_test+batch_test,:] #np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution_out*5:resolution_out*6]
				test_mse,test_mae,test_prediction = sess.run([mse, mae, prediction], feed_dict={xs:x_test,ys:y_test})
				test_total_mse += test_mse
				test_total_mae += test_mae
			
			plt.figure()
			plt.imshow(x_test[0,:].reshape(nelx,nelx),interpolation='None',cmap='gray') 
			plt.savefig('Test_result/draw_dens000.png')
			plt.close()
			      
			plt.figure()
			plt.imshow(y_test[0,:].reshape(height,width),interpolation='None')#,cmap='gray') 
			plt.savefig('Test_result/draw_temp000_exact.png')
			plt.close()
			plt.figure()
			plt.imshow(test_prediction[0,:].reshape(height,width),interpolation='None')#,cmap='gray') 
			plt.savefig('Test_result/draw_temp000_pred.png')
			plt.close()
    
			plt.figure()
			plt.scatter(y_test[0:50,2],test_prediction[0:50,2])
			plt.plot(y_test[0:50,2],y_test[0:50,2],'r-')
			#plt.show()
			plt.savefig("Test_result/test_accuracy_002.png")
			plt.close()
			      
			plt.figure()
			plt.scatter(y_test[0:50,:].reshape(1,50*height*width),test_prediction[0:50,:].reshape(1,50*height*width))
			plt.plot(y_test[0:50,:].reshape(50*height*width),y_test[0:50,:].reshape(50*height*width),'r-')
			#plt.show()
			plt.savefig("Test_result/test_accuracy_all.png")
			plt.close() 
			print('testing relative error=',np.mean(abs(test_prediction[0:50,:].reshape(1,50*height*width)-y_test[0:50,:].reshape(1,50*height*width))/np.maximum(abs(y_test[0:50,:].reshape(1,50*height*width)),1e-2)))    
      	
			mse_test[epoch] = test_total_mse/test_iter_num
			mae_test[epoch] = test_total_mae/test_iter_num
			print('MSE_test:', mse_test[epoch], end = '   ')
			print('MAE_test:', mae_test[epoch])

			if (epoch+1)%10 == 0: #1000
				saver.save(sess, "./savemodel/model")
			current_time = time.localtime()
			print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))
		saver.save(sess, "./savemodel/model")
		print('Training is finished!')	
	end_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
	print('Time end: ', time.strftime('%Y-%m-%d %H:%M:%S', end_time))

if __name__ == '__main__':
	first()
