import tensorflow as tf
sess = tf.InteractiveSession()

input_dim = 200
output_dim = 2

sess = tf.InteractiveSession()

# every placeholder need to be feed by data each time tensorflow run a computation
x = tf.placeholder( tf.float32, shape = [None, input_dim] )
y_ = tf.placeholder( tf.float32, shape = [None, output_dim] )

# weights are defined as variables in tensorflow
# configuration
w = tf.Variable(tf.zeros([input_dim,output_dim]))
b = tf.Variable(tf.zeros([output_dim]))
# initialization
sess.run( tf.global_variables_initializer() )

'''
softmax_cross_entropy_with_logits(y, y_) takes y as input of a softmax layer and calulate the sumation of cross entropy over each dimension
reduce_mean() takes the average over all these sums of samples in the batch.
''' 
y = tf.matmul(x,w)+b;
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y, y_ ) )

# train configuration
'''
learning rate = 0.5
'''
train_step = tf.train.GradientDescentOptimizer( 0.5 ).minimize( cross_entropy )

# train
#  import input data
import numpy as np
samplesPath = './data/aclImdb/train/wvFea.csv'
samples = np.loadtxt( open(samplesPath,'r'), delimiter=' ');
n = samples.shape[0];
label = samples[:,0]
one_hot_label = np.zeros( (n,2) );
for i in range(n):
	one_hot_label[i][0] = label[i];
	one_hot_label[i][1] = 1 - label[i]

import sys
# show the process indicator
def processBar( cur, total ):
	# bar_segment is better to be an divisor of 100
	bar_segment = 20 
	period = total/bar_segment+1
	if cur % period != 0 and cur+1!=total :
		return
	if( cur != 0 ):
		sys.stdout.write('\r')
	p = cur/period
	if( cur+1 == total ):
		p = bar_segment
	sys.stdout.write('[')
	for i in range(p):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(bar_segment-p):
		sys.stdout.write('-')
	sys.stdout.write('] ')
	sys.stdout.write( str(p*100/bar_segment)+'%' )
	if( cur+1 == total ):
		sys.stdout.write('\n')
	sys.stdout.flush()

print samples.shape
batch_sz = 200;
itr = 100000
for i in range(itr):
	idx = np.random.randint( low = 0, high = n-1, size=[batch_sz, 1],)
	batch_x = np.squeeze( samples[idx,1:1+input_dim] )
	batch_y = np.squeeze( one_hot_label[idx,:] )
	#print batch_y
	train_step.run( feed_dict = {x:batch_x, y_:batch_y} )
	processBar( i, itr )
# evaluate
'''
equal return a boolean matrix( typically a vector and each entry denotes a sample) over the entries
argmax return the index of the maximum value along specific dimension( say along columns specified by the second variable 1)
'''
correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )
print accuracy.eval( feed_dict = { x: np.squeeze( samples[:,1:1+input_dim] ), y_: one_hot_label } )

