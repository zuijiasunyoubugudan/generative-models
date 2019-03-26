import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python import debug as tf_debug
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

learning_rate=0.001
nb_epoch=10
testing_iters=10000
batch_size=8
display_step=100
conv1_kernal=32
conv2_kernal=64
latent_dim = 60
n_input=784
n_classes=10
dropout=0.75
dropout2=0.75

def conv2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
	
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lrelu(x, leak=0.01, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def batchnorm(inputs, is_test, iteration, convolutional=True):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) 
    bnepsilon = 1e-4
    if convolutional:
        mean, variance = tf.nn.moments(inputs, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(inputs, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(inputs, m, v, beta, scale, bnepsilon)
    return Ybn, update_moving_averages

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape
l2_loss = tf.constant(0.0)
#lam = tf.Variable(tf.constant(1.0, shape=[]))
lam = 0
with tf.name_scope('input_layer'):
    x=tf.placeholder(tf.float32,[None,n_input])
    y=tf.placeholder(tf.float32,[None,n_classes])
    keep_prob=tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
x_reshape=tf.reshape(x,shape=[-1,28,28,1])

with tf.name_scope('Conv_Encoder_layer1'):
    w_conv_encoder1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.001))
    #w_conv_encoder1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
    b_conv_encoder1 = tf.Variable(tf.constant(0.1, shape=[32]))
    conv_encoder1=tf.nn.conv2d(x_reshape,w_conv_encoder1,strides=[1,2,2,1],padding='SAME')
    conv_encoder1=tf.nn.bias_add(conv_encoder1, b_conv_encoder1)
    #conv_encoder1,update_ema1=batchnorm(conv_encoder1, tst, iter)
    conv_encoder1=tf.nn.relu(conv_encoder1)
    #conv_encoder1=lrelu(conv_encoder1)
    #conv_encoder1=tf.nn.dropout(conv_encoder1, keep_prob, compatible_convolutional_noise_shape(conv_encoder1))
l2_loss += tf.nn.l2_loss(w_conv_encoder1)

with tf.name_scope('Conv_Encoder_layer2'):
    w_conv_encoder2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.001))
    #w_conv_encoder2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
    b_conv_encoder2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv_encoder2=tf.nn.conv2d(conv_encoder1,w_conv_encoder2,strides=[1,2,2,1],padding='SAME')
    conv_encoder2=tf.nn.bias_add(conv_encoder2, b_conv_encoder2)
    #conv_encoder2,update_ema2=batchnorm(conv_encoder2, tst, iter)
    conv_encoder2=tf.nn.relu(conv_encoder2)
    #conv_encoder2=lrelu(conv_encoder2)
    #conv_encoder2=tf.nn.dropout(conv_encoder2, keep_prob, compatible_convolutional_noise_shape(conv_encoder2))
l2_loss += tf.nn.l2_loss(w_conv_encoder2)

with tf.name_scope('Conv_Encoder_layer3'):
    w_conv_encoder3 = tf.Variable(tf.truncated_normal([5,5,64,128],stddev=0.001))
    #w_conv_encoder3 = tf.Variable(tf.truncated_normal([5,5,64,128]))
    b_conv_encoder3 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv_encoder3=tf.nn.conv2d(conv_encoder2,w_conv_encoder3,strides=[1,1,1,1],padding='VALID')
    conv_encoder3=tf.nn.bias_add(conv_encoder3, b_conv_encoder3)
    #conv_encoder3,update_ema3=batchnorm(conv_encoder3, tst, iter)
    conv_encoder3=tf.nn.relu(conv_encoder3)
    #conv_encoder3=lrelu(conv_encoder3)
    #conv_encoder3=tf.nn.dropout(conv_encoder3, keep_prob, compatible_convolutional_noise_shape(conv_encoder3))
    conv_encoder3=tf.reshape(conv_encoder3,[-1,3*3*128])

l2_loss += tf.nn.l2_loss(w_conv_encoder3)

with tf.name_scope('Hidden_Encoder_layer'):
    W_encoder_input_hidden = tf.Variable(tf.truncated_normal([3*3*128,2*latent_dim],stddev=0.001))
    b_encoder_input_hidden = bias_variable([2*latent_dim])
    hidden_encoder = tf.matmul(conv_encoder3, W_encoder_input_hidden) + b_encoder_input_hidden
    #hidden_encoder,update_ema4=batchnorm(hidden_encoder, tst, iter, convolutional=False)
    #hidden_encoder = tf.nn.dropout(hidden_encoder, keep_prob)
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

with tf.name_scope('Mean_Encoder_layer'):
    mu_encoder = hidden_encoder[:, :latent_dim]

with tf.name_scope('Sigma_Encoder_layer'):
    #logvar_encoder = 1e-6 + tf.nn.softplus(hidden_encoder[:, latent_dim:])
    logvar_encoder = hidden_encoder[:, latent_dim:]

with tf.name_scope('Sample_epsilon'):
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

with tf.name_scope('latent_variable'):
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)
    #z_reshape=tf.reshape(z,shape=[-1,2,2,32])

with tf.name_scope('Hidden_Decoder_layer'):
    W_decoder_input_hidden = tf.Variable(tf.truncated_normal([latent_dim,2*2*128],stddev=0.001))
    b_decoder_input_hidden = bias_variable([2*2*128])
    hidden_decoder = tf.matmul(z, W_decoder_input_hidden) + b_decoder_input_hidden
    hidden_decoder = tf.nn.relu(hidden_decoder)
    #hidden_decoder = lrelu(hidden_decoder)
    hidden_decoder = tf.reshape(hidden_decoder,shape=[-1,2,2,128])

# with tf.name_scope('Deconv_Recon_layer1'):
    # w_reconv1 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.001))
    # tf.summary.histogram('Recon_layer1/weights', w_reconv1)
    # b_reconv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    # tf.summary.histogram('Recon_layer1/bias', b_reconv1)
    # conv_recon1=tf.nn.conv2d_transpose(hidden_decoder,w_reconv1,output_shape=[batch_size,7,7,32], strides=[1,1,1,1],padding="VALID")
    # conv_recon1=tf.nn.bias_add(conv_recon1, b_reconv1)
    # conv_recon1=tf.nn.relu(conv_recon1)

# with tf.name_scope('Deconv_Recon_layer2'):
    # w_reconv2 = tf.Variable(tf.truncated_normal([5,5,16,32],stddev=0.001))
    # tf.summary.histogram('Recon_layer2/weights', w_reconv2)
    # b_reconv2 = tf.Variable(tf.constant(0.1, shape=[16]))
    # tf.summary.histogram('Recon_layer2/bias', b_reconv2)
    # conv_recon2=tf.nn.conv2d_transpose(conv_recon1,w_reconv2,output_shape=[batch_size,14,14,16], strides=[1,2,2,1],padding="SAME")
    # conv_recon2=tf.nn.bias_add(conv_recon2, b_reconv2)
    # conv_recon2=tf.nn.relu(conv_recon2)

# with tf.name_scope('Deconv_Recon_layer3'):
    # w_reconv3 = tf.Variable(tf.truncated_normal([5,5,1,16],stddev=0.001))
    # tf.summary.histogram('Recon_layer3/weights', w_reconv3)
    # b_reconv3 = tf.Variable(tf.constant(0.1, shape=[1]))
    # tf.summary.histogram('Recon_layer3/bias', b_reconv3)
    # conv_recon3=tf.nn.conv2d_transpose(conv_recon2,w_reconv3,output_shape=[batch_size,28,28,1], strides=[1,2,2,1],padding="SAME")
    # conv_recon3=tf.nn.bias_add(conv_recon3, b_reconv3)
    # #conv_recon3=tf.nn.relu(conv_recon3)
    # conv_recon3_reshape=tf.reshape(conv_recon3,shape=[-1,784])

with tf.name_scope('Deconv_Decoder_layer1'):
    w_deconv1 = tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.001))
    #w_deconv1 = tf.Variable(tf.truncated_normal([3,3,32,64]))
    b_deconv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv_decoder1=tf.nn.conv2d_transpose(hidden_decoder,w_deconv1,output_shape=[batch_size,3,3,64], strides=[1,2,2,1],padding="SAME")
    conv_decoder1=tf.nn.bias_add(conv_decoder1, b_deconv1)
    #conv_decoder1,update_ema5=batchnorm(conv_decoder1, tst, iter)
    conv_decoder1=tf.nn.relu(conv_decoder1)
    #conv_decoder1=lrelu(conv_decoder1)

with tf.name_scope('Deconv_Decoder_layer2'):
    w_deconv2 = tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.001))
    #w_deconv2 = tf.Variable(tf.truncated_normal([3,3,1,32]))
    b_deconv2 = tf.Variable(tf.constant(0.1, shape=[32]))
    conv_decoder2=tf.nn.conv2d_transpose(conv_decoder1,w_deconv2,output_shape=[batch_size,5,5,32], strides=[1,1,1,1],padding="VALID")
    conv_decoder2=tf.nn.bias_add(conv_decoder2, b_deconv2)
    conv_decoder2=tf.nn.relu(conv_decoder2)
    #conv_decoder2=lrelu(conv_decoder2)

with tf.name_scope('Deconv_Decoder_layer3'):
    w_deconv3 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.001))
    #w_deconv2 = tf.Variable(tf.truncated_normal([3,3,1,32]))
    b_deconv3 = tf.Variable(tf.constant(0.1, shape=[1]))
    conv_decoder3=tf.nn.conv2d_transpose(conv_decoder2,w_deconv3,output_shape=[batch_size,10,10,1], strides=[1,2,2,1],padding="SAME")
    conv_decoder3=tf.nn.bias_add(conv_decoder3, b_deconv3)
	
with tf.name_scope('filter'):
    filter = conv_decoder3

	

with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        for i in range(conv1_kernal/4):
            if i==0:
                filter_tmp = tf.reshape(filter[i],[10,10,1,-1])
                wc1 = filter_tmp[0:5,0:5,:,:]
                wc1 = tf.concat([wc1,filter_tmp[0:5,5:10,:,:]],3)
                wc1 = tf.concat([wc1,filter_tmp[5:10,0:5,:,:]],3)
                wc1 = tf.concat([wc1,filter_tmp[5:10,5:10,:,:]],3)
            else:
                filter_tmp = tf.reshape(filter[i],[10,10,1,-1])
                wc1 = tf.concat([wc1,filter_tmp[0:5,0:5,:,:]],3)
                wc1 = tf.concat([wc1,filter_tmp[0:5,5:10,:,:]],3)
                wc1 = tf.concat([wc1,filter_tmp[5:10,0:5,:,:]],3)
                wc1 = tf.concat([wc1,filter_tmp[5:10,5:10,:,:]],3)
        tf.summary.histogram('conv1/weights', wc1)
    with tf.name_scope('bias'):
        bc1 = tf.Variable(tf.constant(0.1, shape=[conv1_kernal]))
        tf.summary.histogram('conv1/bias', bc1)
    conv1=conv2d(x_reshape,wc1,bc1)
    conv1=maxpool2d(conv1,k=2)

with tf.name_scope('reconstruction'):
    with tf.name_scope('weights'):
        wc_recon = tf.Variable(tf.truncated_normal([5,5,1,conv1_kernal]))
    with tf.name_scope('bias'):
        bc_recon = tf.Variable(tf.constant(0.1, shape=[1]))
    recon=tf.nn.conv2d_transpose(conv1,wc_recon,output_shape=[batch_size,28,28,1], strides=[1,2,2,1],padding="SAME")
    recon=tf.nn.bias_add(recon, bc_recon)
    #recon=tf.nn.relu(recon)
    recon_reshape=tf.reshape(recon,shape=[-1,784])

with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
        wc2 = tf.Variable(tf.truncated_normal([5,5,conv1_kernal,conv2_kernal]))
        tf.summary.histogram('conv2/weights', wc2)
    with tf.name_scope('bias'):
        bc2 = tf.Variable(tf.constant(0.1, shape=[conv2_kernal]))
        tf.summary.histogram('conv2/bias', bc2)
    conv2=conv2d(conv1,wc2,bc2)
    conv2=maxpool2d(conv2,k=2)
	
with tf.name_scope('fc'):
    with tf.name_scope('weights'):
        wd1 = tf.Variable(tf.truncated_normal([7*7*conv2_kernal,1024]))
    with tf.name_scope('bias'):
        bd1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    fc1=tf.reshape(conv2,[-1,wd1.get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,wd1),bd1)
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,keep_prob)
		
with tf.name_scope('out'):
    with tf.name_scope('weights'):
        wout = tf.Variable(tf.truncated_normal([1024,n_classes]))
    with tf.name_scope('bias'):
        bout = tf.Variable(tf.constant(0.1, shape=[n_classes]))      
    out_tmp1=tf.add(tf.matmul(fc1,wout),bout)
    out_tmp2=tf.nn.softmax(out_tmp1)

#update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5)
with tf.name_scope('KL_loss'):
    KLD = 0.5 * tf.reduce_sum( tf.exp(logvar_encoder) + tf.pow(mu_encoder, 2) - logvar_encoder - 1, reduction_indices=1)
    KLD = tf.reduce_mean(KLD)
with tf.name_scope('class_loss'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_tmp1,labels=y))
with tf.name_scope('recon_loss'):
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=recon_reshape, labels=x), reduction_indices=1) 
    recon_loss = tf.reduce_mean(recon_loss)
    tf.summary.scalar("reconstruction_loss", recon_loss)
# with tf.name_scope('filter_recon_loss'):
    # filter_recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv_recon3_reshape, labels=x), reduction_indices=1) 
    # filter_recon_loss = tf.reduce_mean(filter_recon_loss)
    # tf.summary.scalar("filter_reconstruction_loss", filter_recon_loss)
with tf.name_scope('train'):
    loss=cost+KLD+recon_loss+lam*l2_loss
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('train_loss',loss)
with tf.name_scope('accuracy'):
    correct_pred=tf.equal(tf.argmax(out_tmp2,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(allow_growth=True)
test_acc_count=[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for i in range(nb_epoch):
        print 'Epoch:'+str(i)+'/'+str(nb_epoch)
        for j in range((55000/batch_size)+1):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            if j % display_step == 0:
                los, acc = sess.run([loss, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.0,tst:False})
                print("epoch: " + str(i) + ", batch: " + str(j) + ", Accuracy= " + "{:.5f}".format(acc))
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout,tst:False})
            #sess.run(update_ema, feed_dict={x:batch_x,y:batch_y,keep_prob:1.0,tst:False,iter:j})
    saver.save(sess, './var.ckpt')
    print("optimizer finished")
    i=0
    accuracy_sum=0
    while (i+1)*batch_size<=testing_iters:
        accuracy_tmp=sess.run(accuracy,feed_dict={x:mnist.test.images[i*batch_size:(i+1)*batch_size],y:mnist.test.labels[i*batch_size:(i+1)*batch_size],keep_prob: 1.0,tst:True})
        accuracy_sum+=accuracy_tmp
        i+=1
    print("testing accuracy:",accuracy_sum/1250)

                                                          
    
