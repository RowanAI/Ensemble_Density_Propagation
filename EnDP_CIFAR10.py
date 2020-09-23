### Copyright (C) <2020>  <Giuseppina Carannante, Dimah Dera , Ghulam Rasool, Nidhal Bouaynaya, Lyudmila Mihaylova>
### Paper : ROBUST LEARNING VIA ENSEMBLE DENSITY PROPAGATION IN DEEP NEURAL NETWORKS
### IEEE International Workshop on Machine Learning for Signal Processing, Sept. 2020.

import tensorflow as tf
import pickle
import sys
import os
import time
import numpy as np
import glob
import cv2
#from numpy import linalg as LA
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import timeit
from scipy.misc import toimage
from Adding_noise import random_noise
from Covariance_propagation import first_convolution,intermediate_convolution,Ensemble_activation, intermediate_convolution_approx, activation, max_pooling, fully_connected
from scipy.ndimage import rotate
plt.ioff()
from sklearn.model_selection import train_test_split
#%matplotlib inline
import tensorflow_probability as tfp
tfd = tfp.distributions

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# classes name which are given in CIFAR-10 dataset(10 types of images in this dataset which are given below)
class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#this function gives you predicted class name by CNN with highest probabilistic class
def classify_name(predicts):
    max =predicts[0,0]
    temp =0
    for i in range(len(predicts[0])):
        #check higher probable class 
        if predicts[0,i]>max:
                max = predicts[0,i]
                temp = i
    # print higher probale class name
    print(class_name[temp])
 

def update_progress(progress):
    barLength = 10 
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "VALID") #VALID

def conv2d_w_pad(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

    
def load_cifar10_data():
    Y_train = np.zeros((50000,10),dtype = np.float32)
    Y_valid = np.zeros((5000,10), dtype = np.float32)
    Y_test = np.zeros((5000,10),dtype = np.int32)
    
    data1 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_1")
    data2 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_2")
    data3 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_3")
    data4 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_4")
    data5 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_5")
    
    
    labels1 = data1[b'labels']
    data1 = data1[b'data'] * 1.0
    labels2 = data2[b'labels']
    data2 = data2[b'data'] * 1.0
    labels3 = data3[b'labels']
    data3 = data3[b'data'] * 1.0
    labels4 = data4[b'labels']
    data4 = data4[b'data']  * 1.0
    labels5 = data5[b'labels']
    data5 = data5[b'data']  * 1.0    
    
    X_tr = np.concatenate([data1, data2, data3, data4, data5], axis=0)
    X_tr = np.dstack((X_tr[:, :1024], X_tr[:, 1024:2048], X_tr[:, 2048:])) / 1.0
    X_tr = (X_tr - 128) / 255.0
    X_tr = X_tr.reshape(-1, 32, 32, 3)    
    y_tr = np.concatenate([labels1, labels2, labels3, labels4, labels5], axis=0)    
   
    num_classes = len(np.unique(y_tr))    
    print("X_tr", X_tr.shape)
    print("y_tr", y_tr.shape)    
    # import the test data
    test_data = unpickle("./cifar-10-python/cifar-10-batches-py/test_batch")    
    X_test = test_data[b'data']
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:])) / 1.0
    X_test = (X_test - 128) / 255.0
    X_test = X_test.reshape(-1, 32, 32, 3)
    y_test = np.asarray(test_data[b'labels'])    
    # split into test and validation
    X_te, X_cv, y_te, y_cv = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    for i in range(50000):
        a = y_tr[i]
        Y_train[i,a] = 1        
    for i in range(5000):
        a = y_cv[i]
        Y_valid[i,a] = 1
    for i in range(5000):
        a = y_te[i]
        Y_test[i,a] = 1    
    return X_tr, Y_train, X_cv, Y_valid, X_te, Y_test


def get_batches(X, y, batch_size, crop=False, distort=False, shuffle=True):
    # Shuffle X,y
    if shuffle:
        shuffled_idx = np.arange(len(y))
        np.random.shuffle(shuffled_idx)
    i, h, w, c = X.shape    
    
    for i in range(0, len(y)- batch_size + 1, batch_size):
        if shuffle:
            batch_idx = shuffled_idx[i : i + batch_size]
        else:
            batch_idx = slice(i, i + batch_size)
        X_return = X[batch_idx]        
        # optional random crop of images
        if crop:
            woff = (w - 24) // 4
            hoff = (h - 24) // 4
            startw = np.random.randint(low=woff,high=woff*2)
            starth = np.random.randint(low=hoff,high=hoff*2)
            X_return = X_return[:,startw:startw+24,starth:starth+24,:]       
        # do random flipping of images
        coin = np.random.binomial(1, 0.5, size=None)
        if coin and distort:
            X_return = X_return[...,::-1,:]        
        yield X_return, y[batch_idx]
        
# Function to propagate the mean and covariance matrix of the variational distribution through the convolutional neural network. 
# Inputs:
#       x: the input image (tensor) from the dataset
#       w_mean: the variable with the mean tensor of the convolutional kernels, of fully-connected layer(FC) weights, and of the bias term in FC
#       s: the variable with the variance tensor of the convolutional kernels, of the FC weights, and of the bias term in FC
#       new_size/new_size2/new_size3: the size of the feature map after the max-pooling layers # ((image_size - filter_size + 2P)/stride)+1) 
#       keep_prob1/keep_prob2/keep_prob3: the probability for dropouts
#       num_filters: the number of kernels (filters) in the convolutional layers 
#       image_size: the size of the input image 
#       patch_size: the size of the convolutional kernels 
#       num_channel:the number of channels in the input image. (1 for MNIST data as the dataset are grayscale images)
#       num_labels: the number of classes in the dataset. It is 10 for the MNIST & CIFAR-10 dataset
#       epsilon_std = standard deviation for random initialization
# Outputs: 

#       mu_y, y_out: The output and mean vector (prediction - class scores).
#       f_fc1, mu_f_fc1: output FC
#       sigma_y: The output covariance matrix (predictive covariance matrix).
#       sigma_f: The covariance matrix before the soft-max layer. 
#       target_conv_layer1,target_conv_layer_grad1, 
#       target_conv_layer2, target_conv_layer_grad2: last convolutional layer gradients for generating gradCAM visualization.
       
def Model_with_uncertainty_computation(x, y_label,w_mean, s, new_size,  new_size2, new_size3, keep_prob1, keep_prob2, keep_prob3, num_filters,
            image_size=32, patch_size=3, num_channel=3, num_labels=10, epsilon_std= 1.0):
    
    conv1_weight_epsilon = tf.random_normal([patch_size, patch_size,num_channel,num_filters[0]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv1= w_mean['m1'] + tf.multiply(tf.log(1. + tf.exp(s['s1'])), conv1_weight_epsilon)    
    mu_z = conv2d(x, w_mean['m1'])# shape=[1, image_size,image_size,num_filters[0]]
    sigma_z = first_convolution(x , s['s1'] , num_filters[0], patch_size,num_channel, pad="VALID" ) #shape=[num_filter,image_size*image_size, image_size*image_size]
    ######################################################   
    # propagation through the activation function  
    z = conv2d(x, W_conv1)#shape=[1, image_size,image_size,num_filters[0]]
    image_size = image_size - patch_size + 1

    g = tf.nn.elu(z)#shape=[1, image_size,image_size,num_filters[0]]    
    mu_g ,sigma_g = Ensemble_activation(mu_z, sigma_z, image_size, num_filters[0])
    
    g = tf.nn.dropout(g, keep_prob1) 
    mu_g = tf.nn.dropout(mu_g, keep_prob1)    
    ######################################################    
    conv2_weight_epsilon = tf.random_normal([patch_size, patch_size,num_filters[0],num_filters[1]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv2= w_mean['m2'] + tf.multiply(tf.log(1. + tf.exp(s['s2'])), conv2_weight_epsilon)
    z2 = conv2d(g, W_conv2)#shape=[1, image_size,image_size,num_filters[1]]
    mu_z2 = conv2d(mu_g, w_mean['m2'])# shape=[1, image_size,image_size,num_filters[1]]
    new_im_size = image_size - patch_size + 1
    sigma_z2 = intermediate_convolution_approx(w_mean['m2'],s['s2'],mu_g,sigma_g,patch_size,num_filters[0],num_filters[1], image_size, new_im_size, pad="VALID")
    image_size = image_size - patch_size + 1
    ######################################################         

    g2 = tf.nn.elu(z2)#shape=[1, image_size,image_size,num_filters[1]]   
   
    mu_g2 ,sigma_g2 = Ensemble_activation(mu_z2, sigma_z2, image_size, num_filters[1])  
    ######################################################
    # propagation through the pooling layer    
    p = max_pool_2x2(g2)  #shape=[1, new_size,new_size,num_filters[0]]
    mu_p , argmax_p = tf.nn.max_pool_with_argmax(mu_g2 , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #shape=[1, new_size,new_size,num_filters[1]]
    sigma_p = max_pooling(sigma_g2, argmax_p, num_filters[1], new_size, image_size )

    p = tf.nn.dropout(p, keep_prob1) 
    mu_p = tf.nn.dropout(mu_p, keep_prob1)
    ######################################################  
    conv3_weight_epsilon = tf.random_normal([patch_size, patch_size,num_filters[1],num_filters[2]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv3= w_mean['m3'] + tf.multiply(tf.log(1. + tf.exp(s['s3'])), conv3_weight_epsilon)    
    z3 = conv2d(p, W_conv3)#shape=[1, new_size,new_size,num_filters[2]]
    mu_z3 = conv2d(mu_p, w_mean['m3'])# shape=[1, new_size,new_size,num_filters[2]]
    new_im_size = new_size - patch_size + 1
    sigma_z3 = intermediate_convolution_approx(w_mean['m3'],s['s3'],mu_p,sigma_p,patch_size,num_filters[1],num_filters[2], new_size, new_im_size, pad="VALID")
    new_size = new_size - patch_size + 1    
    ######################################################  

    g3 = tf.nn.elu(z3)#shape=[1, new_size,new_size,num_filters[2]]     
    mu_g3 ,sigma_g3 = Ensemble_activation(mu_z3, sigma_z3, new_size, num_filters[2])  

    g3 = tf.nn.dropout(g3, keep_prob2) 
    mu_g3 = tf.nn.dropout(mu_g3, keep_prob2)
    ###################################################### 
    conv4_weight_epsilon = tf.random_normal([patch_size, patch_size,num_filters[2],num_filters[3]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv4= w_mean['m4'] + tf.multiply(tf.log(1. + tf.exp(s['s4'])), conv4_weight_epsilon)
    z4 = conv2d_w_pad(g3, W_conv4)#shape=[1, new_size,new_size,num_filters[3]]
    mu_z4 = conv2d_w_pad(mu_g3, w_mean['m4'])# shape=[1, new_size,new_size,num_filters[3]]    
    sigma_z4 = intermediate_convolution_approx(w_mean['m4'],s['s4'],mu_g3,sigma_g3,patch_size,num_filters[2],num_filters[3], new_size, new_size, pad="SAME")

    ######################################################     

    g4 = tf.nn.elu(z4)#shape=[1, new_size,new_size,num_filters[3]]      
    mu_g4 ,sigma_g4 = Ensemble_activation(mu_z4, sigma_z4, new_size, num_filters[3])  
    ######################################################
    p2 = max_pool_2x2(g4)  #shape=[1, new_size2,new_size2,num_filters[3]]
    mu_p2, argmax_p2 = tf.nn.max_pool_with_argmax(mu_g4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #shape=[1, new_size2,new_size2,num_filters[3]]
    sigma_p2 = max_pooling(sigma_g4, argmax_p2, num_filters[3], new_size2, new_size )          
  
    ###################################################### 
    conv5_weight_epsilon = tf.random_normal([patch_size, patch_size,num_filters[3],num_filters[4]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv5 = w_mean['m5'] + tf.multiply(tf.log(1. + tf.exp(s['s5'])), conv5_weight_epsilon)

    z5 = conv2d_w_pad(p2, W_conv5)#shape=[1, new_size2,new_size2,num_filters[4]]
    mu_z5 = conv2d_w_pad(mu_p2, w_mean['m5'])# shape=[1, new_size2,new_size2,num_filters[4]]        
    sigma_z5 = intermediate_convolution_approx(w_mean['m5'],s['s5'],mu_p2,sigma_p2,patch_size,num_filters[3],num_filters[4], new_size2, new_size2, pad="SAME")

    ######################################################    

    g5 = tf.nn.elu(z5)#shape=[1, new_size2,new_size2,num_filters[4]]       
    mu_g5 ,sigma_g5 = Ensemble_activation(mu_z5, sigma_z5, new_size2, num_filters[4])    

    g5 = tf.nn.dropout(g5, keep_prob1) 
    mu_g5 = tf.nn.dropout(mu_g5, keep_prob1)  
    ###################################################### 
    conv6_weight_epsilon = tf.random_normal([patch_size, patch_size,num_filters[4],num_filters[5]], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)   
    W_conv6 = w_mean['m6'] + tf.multiply(tf.log(1. + tf.exp(s['s6'])), conv6_weight_epsilon)
    z6 = conv2d_w_pad(g5, W_conv6)#shape=[1, new_size2,new_size2,num_filters[5]]
    mu_z6 = conv2d_w_pad(mu_g5, w_mean['m6'])# shape=[1, new_size2,new_size2,num_filters[5]]    
    sigma_z6 = intermediate_convolution_approx(w_mean['m6'],s['s6'],mu_g5,sigma_g5,patch_size,num_filters[4],num_filters[5], new_size2, new_size2, pad="SAME")
    ######################################################    

    g6 = tf.nn.elu(z6)#shape=[1, new_size2,new_size2,num_filters[5]]      
    mu_g6 ,sigma_g6 = Ensemble_activation(mu_z6, sigma_z6, new_size2, num_filters[5]) 
    ######################################################
    p3 = max_pool_2x2(g6)  #shape=[1, new_size3,new_size3,num_filters[5]]
    mu_p3, argmax_p3 = tf.nn.max_pool_with_argmax(mu_g6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #shape=[1, new_size3,new_size3,num_filters[5]]
    sigma_p3 = max_pooling(sigma_g6, argmax_p3, num_filters[5], new_size3, new_size2 )  
    ######################################################
    # # propagation through the Fully Connected

    b = tf.reshape(p3, [-1, new_size3*new_size3*num_filters[5]]) #shape=[1, new_size3*new_size3*num_filters[5]]
    mu_b = tf.reshape(mu_p3, [-1, new_size3*new_size3*num_filters[5]]) #shape=[1, new_size3*new_size3*num_filters[5]] 
    mu_b = tf.nn.dropout(mu_b, keep_prob3)      
    ######################################################
    # # propagation through the Fully Connected        
    fc1_weight_epsilon = tf.random_normal([new_size3*new_size3*num_filters[5], num_labels], mean=0.0, stddev=epsilon_std, dtype=tf.float32, seed=None, name=None)
    fc1_bias_epsilon = tf.random_normal([num_labels], mean=0.0, stddev=epsilon_std*0.0001, dtype=tf.float32, seed=None, name=None)
    
    W_fc1 = w_mean['m7'] + tf.multiply(tf.log(1. + tf.exp(s['s7'])), fc1_weight_epsilon)
    b_fc1 = w_mean['m8'] + tf.multiply(tf.log(1. + tf.exp(s['s8'])) , fc1_bias_epsilon)  
    
    f_fc1 = tf.matmul(b, W_fc1) + b_fc1 #shape=[1, num_labels]
    mu_f_fc1 = tf.matmul(mu_b, w_mean['m7']) + w_mean['m8'] #shape=[1, num_labels]
    sigma_f = fully_connected(w_mean['m7'] ,s['s7'], mu_b, sigma_p3, num_filters[5], new_size3, num_labels )   
    ######################################################  
    y_out = tf.nn.softmax(f_fc1 ) #shape=[1, num_labels]

    
    mu_y = tf.nn.softmax(mu_f_fc1) #shape=[1, num_labels]
    # compute the gradient of softmax manually
    grad_f1 = tf.matmul(tf.transpose(mu_y), mu_y)  
    diag_f = tf.diag(tf.squeeze(mu_y))
    grad_soft = diag_f - grad_f1 #shape=[num_labels, num_labels] #tf.diag_part    
    sigma_y = tf.matmul(grad_soft,   tf.matmul(sigma_f, grad_soft, transpose_b=True))#shape=[num_labels,num_labels] 
    
    y_label_c1 = tf.reduce_sum(tf.multiply(mu_f_fc1 ,  y_label), axis=1)
    y_label_c2 = tf.reduce_sum(tf.multiply(f_fc1 ,  y_label), axis=1)
    # Get last convolutional layer gradient for generating gradCAM visualization
    target_conv_layer1 = mu_p3
    target_conv_layer2 = p3   
    target_conv_layer_grad1 = tf.gradients(y_label_c1, target_conv_layer1)[0]
    target_conv_layer_grad2 = tf.gradients(y_label_c2, target_conv_layer2)[0]
    return y_out, mu_y, f_fc1, mu_f_fc1, sigma_y, sigma_f, target_conv_layer1, target_conv_layer_grad1, target_conv_layer2, target_conv_layer_grad2


# Visualization function: 
# It visualize nine images from the test set with corresponding true and predicted value.
def plot_images(images, sigma_std, epoch, cls_true,path, cls_pred=None, smooth=True, noise=0.0):
    assert len(images) == len(cls_true) == 9
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'bicubic'#'nearest'            
        image = images[i, :, :, :]         
        image += noise  # Add the adversarial noise to the image.       
        image1 = image
        # Plot image.
        ax.imshow(toimage(image1), interpolation=interpolation)         
        # Name of the true class.
        cls_true_name = class_name[cls_true[i]]
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_name[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)        
        ax.set_xticks([])
        ax.set_yticks([])     
 
    plt.savefig(os.path.join(path, "EnVI_CNN_on_CIFAR10_Tested_images.png"))   
    plt.close(fig)

# KL -divergence for convolution layers.
def kl_divergence_conv(patch_size, num_filters, conv_weight_M, conv_weight_sigma ):    
    c_s = tf.log(1. + tf.exp(conv_weight_sigma))
    kl_loss_conv = - 0.5 * tf.reduce_mean(patch_size*patch_size + patch_size*patch_size* tf.log(c_s) -
   tf.reduce_sum(tf.abs(conv_weight_M)) - patch_size*patch_size* c_s, axis=-1)  
    return kl_loss_conv
#KL-divergence for FC layers.
def kl_divergence_fc(new_size, num_filters, fc_weight_mu, fc_weight_sigma ):
    f_s = tf.log(1. + tf.exp(fc_weight_sigma))
    kl_loss_fc = - 0.5 * tf.reduce_mean(new_size*new_size*num_filters + new_size*new_size*num_filters* tf.log(f_s)
    - tf.reduce_sum(tf.abs(fc_weight_mu)) - new_size*new_size* num_filters*f_s, axis=-1)   
  
    return kl_loss_fc

# The log-likelihood of the objective function.
# Inputs: 
#       y_pred_mean: The output mean vector (predictive mean).
#       y_pred_sd: The output covariance matrix (predictive covariance matrix).
#       y_test: The ground truth prediction vector
#       num_labels: the number of classes in the dataset. It is 10 for the MNIST dataset
# Output:
#       the expected log-likelihood term of the objective function. 

def nll_gaussian(y_pred_mean,y_pred_sd,y_test, num_labels=10):
    NS = tf.diag(tf.constant(1e-3, shape=[num_labels]))
    I = tf.eye(num_labels)
    y_pred_sd_ns = y_pred_sd + NS
    y_pred_sd_inv = tf.matrix_solve(y_pred_sd_ns, I)
    mu_ = y_pred_mean - y_test
    mu_sigma = tf.matmul(mu_ ,  y_pred_sd_inv) 
    ms = 0.5*tf.matmul(mu_sigma , mu_, transpose_b=True) + 0.5*tf.linalg.slogdet(y_pred_sd_ns)[1]
    ms = tf.reduce_mean(ms)
    return(ms)

def main_function(image_size=32, num_channel=3, patch_size=3, num_filter=[32, 32, 64,64,128,128],num_labels=10,
        batch_size=50, noise_limit=0.01, noise_l2_weight=0.01, adversary_target_cls=3, init_sigma_std=-4.6,
        weight_std=0.01, epochs=25,Adversarial_noise=False, Random_noise=False, gaussain_noise_std=0.00028, Training=True, continue_train=False, load_model=False):     
    init_std = 0.01
     
    gaussain_noise_var = gaussain_noise_std**2
    saved_result_path = './EnVI_CIFAR10_with_sigma_{}/epochs_{}/'.format(init_std, epochs)#
    if Adversarial_noise:
        saved_result_path = './EnVI_Adv_results/EnVI_CIFAR10_with_sigma_{}_adv_{}/epochs_{}/'.format(init_std, noise_limit, epochs)#
        
           
    x = tf.placeholder(tf.float32, shape = (1, image_size,image_size,num_channel), name='x') 
    y = tf.placeholder(tf.float32, shape = (1,num_labels), name='y_true')        
    y_true_cls = tf.argmax(y, axis=1) 
    keep_prob1  = tf.placeholder(tf.float32)
    keep_prob2  = tf.placeholder(tf.float32)
    keep_prob3  = tf.placeholder(tf.float32)
  
    image_size1 = image_size - patch_size+1
    image_size2 = image_size1 - patch_size+1
    new_size = np.int(np.ceil(np.float(image_size2)/2))# ((input_size-filter_size+2P)/stride)+1
    new_size1 = new_size - patch_size+1
    new_size2 = np.int(np.ceil(np.float(new_size1)/2))
    new_size3 = np.int(np.ceil(np.float(new_size2)/2))
    #######################################################   
    if continue_train:    
        if Adversarial_noise:
            file11 = open('./EnVI_Adv_results/EnVI_CIFAR10_with_sigma_{}_adv_{}/epochs_{}/BayesCNN_weights_features.pkl'.format(init_std,noise_limit, 5), 'rb')
        else:    
            file11 = open('./EnVI_CIFAR10_with_sigma_{}/epochs_{}/BayesCNN_weights_features.pkl'.format(init_std, 25), 'rb')        
        con_m1,con_m2,con_m3,con_m4,con_m5,con_m6,fc_m,b_m,conv_s1,conv_s2,conv_s3,conv_s4,conv_s5,conv_s6,fc_s,b_s =   pickle.load(file11) 
        file11.close()
        weights_mean = {
                       'm1': tf.Variable(tf.constant(con_m1, shape=[patch_size,patch_size,num_channel,num_filter[0]],dtype=tf.float32)),
                       'm2': tf.Variable(tf.constant(con_m2, shape=[patch_size,patch_size,num_filter[0],num_filter[1]],dtype=tf.float32)),
                       'm3': tf.Variable(tf.constant(con_m3, shape=[patch_size,patch_size,num_filter[1],num_filter[2]],dtype=tf.float32)),
                       'm4': tf.Variable(tf.constant(con_m4, shape=[patch_size,patch_size,num_filter[2],num_filter[3]],dtype=tf.float32)),
                       'm5': tf.Variable(tf.constant(con_m5, shape=[patch_size,patch_size,num_filter[3],num_filter[4]],dtype=tf.float32)),
                       'm6': tf.Variable(tf.constant(con_m6, shape=[patch_size,patch_size,num_filter[4],num_filter[5]],dtype=tf.float32)),
                       'm7': tf.Variable(tf.constant(fc_m,   shape=[new_size3*new_size3*num_filter[5],num_labels],dtype=tf.float32)),
                       'm8': tf.Variable(tf.constant(b_m,    shape=[num_labels]))        
                    }  
        sigmas = {
                       's1': tf.Variable(tf.constant(conv_s1, shape=[num_filter[0]])),       
                       's2': tf.Variable(tf.constant(conv_s2, shape=[num_filter[1]])),      
                       's3': tf.Variable(tf.constant(conv_s3, shape=[num_filter[2]])),    
                       's4': tf.Variable(tf.constant(conv_s4, shape=[num_filter[3]])),    
                       's5': tf.Variable(tf.constant(conv_s5, shape=[num_filter[4]])),    
                       's6': tf.Variable(tf.constant(conv_s6, shape=[num_filter[5]])),        
                       's7': tf.Variable(tf.constant(fc_s, shape=[num_labels])),
                       's8': tf.Variable(tf.constant(0.0, shape=[num_labels])) 
                    }
    else:
        weights_mean = {
                       'm1': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channel,num_filter[0]],stddev = weight_std)),
                       'm2': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_filter[0],num_filter[1]],stddev = weight_std)),
                       'm3': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_filter[1],num_filter[2]],stddev = weight_std)),
                       'm4': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_filter[2],num_filter[3]],stddev = weight_std)),
                       'm5': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_filter[3],num_filter[4]],stddev = weight_std)),
                       'm6': tf.Variable(tf.truncated_normal([patch_size,patch_size,num_filter[4],num_filter[5]],stddev = weight_std)),
                       'm7': tf.Variable(tf.truncated_normal([new_size3*new_size3*num_filter[5],num_labels],stddev = weight_std)),
                       'm8': tf.Variable(tf.constant(0.0, shape=[num_labels]))        
                   }   
        sigmas = {
                      's1': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[0]])),       
                      's2': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[1]])),      
                      's3': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[2]])),    
                      's4': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[3]])),    
                      's5': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[4]])),    
                      's6': tf.Variable(tf.constant(init_sigma_std, shape=[num_filter[5]])),        
                      's7': tf.Variable(tf.constant(init_sigma_std, shape=[num_labels])),
                      's8': tf.Variable(tf.constant(0.0, shape=[num_labels])) 
                  }

    if Adversarial_noise:
        ADVERSARY_VARIABLES = 'adversary_variables'
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]
        x_noise = tf.Variable(tf.zeros([image_size, image_size, num_channel]),  name='x_noise', trainable=False, collections=collections)
        x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit,    noise_limit))
        x_noisy_image = x + x_noise  
        x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)  
        print('Call the model ....')
        network_out,prediction,class_score1,class_score2,output_sigma,sigma_f,maxmu,maxmu_g,maxh,maxh_g=Model_with_uncertainty_computation(x_noisy_image, y,weights_mean, sigmas,new_size,new_size2,new_size3, keep_prob1, keep_prob2, keep_prob3, num_filter)
        adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)  
        l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)  
    else:
        print('Call the model ....')
        network_out,prediction,class_score1,class_score2,output_sigma,sigma_f,maxmu,maxmu_g,maxh,maxh_g=Model_with_uncertainty_computation(x,y, weights_mean,sigmas,new_size,new_size2,new_size3, keep_prob1, keep_prob2, keep_prob3, num_filter)       
    ######################################################     
    # KL-divergence    
    kl_loss_conv1 = kl_divergence_conv(patch_size, num_filter[0], weights_mean['m1'], sigmas['s1'] )   
    kl_loss_conv2 = kl_divergence_conv(patch_size, num_filter[1], weights_mean['m2'], sigmas['s2'] )      
    kl_loss_conv3 = kl_divergence_conv(patch_size, num_filter[2], weights_mean['m3'], sigmas['s3'] )     
    kl_loss_conv4 = kl_divergence_conv(patch_size, num_filter[3], weights_mean['m4'], sigmas['s4'] )    
    kl_loss_conv5 = kl_divergence_conv(patch_size, num_filter[4], weights_mean['m5'], sigmas['s5'] )     
    kl_loss_conv6 = kl_divergence_conv(patch_size, num_filter[5], weights_mean['m6'], sigmas['s6'] ) 
    kl_loss_fc1   = kl_divergence_fc(new_size3, num_filter[5], weights_mean['m7'], sigmas['s7'] ) 
    ######################################################          
    output_sigma1 = tf.clip_by_value(t=(output_sigma) , clip_value_min=tf.constant(1e-10), 
                                   clip_value_max=tf.constant(1e+10)) # 
    y_pred_cls = tf.argmax(prediction, axis=1)
    N = 50000

    tau = 0.000001 # / n_batches
    print('Compute Cost Function ....')  
    log_likelihood = nll_gaussian(prediction, output_sigma1, y) 
    loss = log_likelihood +  tau *(kl_loss_conv1+kl_loss_conv2+kl_loss_conv3+kl_loss_conv4+kl_loss_conv5+kl_loss_conv6+kl_loss_fc1)#+ beta * regularizers 
    print('Compute Optm ....')
    optm = tf.train.AdamOptimizer(learning_rate = 0.00002).minimize(loss) 
    print('Compute Accuracy ....')
    corr = tf.equal(tf.argmax(prediction, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))   
    if Adversarial_noise:         
        loss_adversary = log_likelihood + l2_loss_noise
        adv_lr = 0.00001
        optimizer_adversary = tf.train.AdamOptimizer(learning_rate= adv_lr ).minimize(loss_adversary, var_list=adversary_variables)  
    print('Initialize Variables ....')
    
    init = tf.global_variables_initializer() 
    saver = tf.train.Saver()
    prob1 = 1.0
    prob2 = 1.0
    prob3 = 0.95
     
    print("loading dataset...")   
    X_train,y_train,X_val, y_val,X_test,y_test = load_cifar10_data() 
    
   
    if not os.path.exists(os.path.join(saved_result_path, "model.meta")):         
        sess = tf.Session()
        if load_model:
            if Adversarial_noise:
                saver.restore(sess, tf.train.latest_checkpoint('./EnVI_Adv_results/EnVI_CIFAR10_with_sigma_{}_adv_{}/epochs_{}'.format(init_std,noise_limit, 5))) 
            else:
                saver.restore(sess, tf.train.latest_checkpoint('./EnVI_CIFAR10_with_sigma_{}/epochs_{}'.format(init_std, 25)))
        else:
            sess.run(init)
        
        if Adversarial_noise:
            sess.run(tf.variables_initializer([x_noise]))        
        start = timeit.default_timer()    
        print("Starting training ....")             
        train_acc = np.zeros(epochs)             
        valid_acc = np.zeros(epochs)               
        for k in range(epochs): 
            print(k+1 ,'/', epochs)
            acc1 = 0
            acc_valid1 = 0                          
            train_batches = 0                     
            val_batches = 0
            tr_co = 0  
            val_co = 0
            for tr_minibatch in get_batches(X_train, y_train, batch_size, crop=False, distort=False):            
                update_progress(train_batches/int(N/batch_size))    
                inputs, targets = tr_minibatch                 
                acc2 = 0
                err2 = 0
                for i in range(batch_size):                          
                    xx_ = np.expand_dims(inputs[i,:,:,:],axis=0) 
                    yy_ = np.expand_dims(targets[i,:], axis=0)                                      
                    sess.run([optm],feed_dict = {x: xx_, y: yy_, keep_prob1 : prob1, keep_prob2 : prob2, keep_prob3 : prob3}) 
                    if (train_batches % 100 == 0) or (train_batches == (int(N/batch_size) - 1)):               
                        acc = sess.run(accuracy,feed_dict = {x: xx_, y: yy_, keep_prob1 : 1., keep_prob2 : 1., keep_prob3 : 1.})                 
                        acc2 += acc
                        tr_co +=1

                    if (train_batches % 200 == 0) or (train_batches == (int(N/batch_size) - 1)):
                        err = sess.run(loss, feed_dict = {x: xx_, y: yy_, keep_prob1 : 1., keep_prob2 : 1., keep_prob3 : 1.})                 
                        err2 += err                        
                if (train_batches % 100 == 0) or (train_batches == (int(N/batch_size) - 1)): 
                    print('Train Acc:', acc2/batch_size,'Train err:', err2/batch_size)                  
                acc1 += acc2             
                train_batches += 1                              
            train_acc[k] = acc1 /  (tr_co)          
            for va_minibatch in get_batches(X_val, y_val, batch_size, crop=False, distort=False):
                update_progress(val_batches/int(5000/batch_size)) 
                inputs, targets = va_minibatch                    
                acc_valid2 = 0
                err_valid2 = 0
                for i in range(batch_size):                   
                    valid_xx_ = np.expand_dims(inputs[i,:,:,:],axis=0) 
                    valid_yy_ = np.expand_dims(targets[i,:], axis=0)                                      
                    sess.run([optm],feed_dict = {x: valid_xx_, y: valid_yy_ , keep_prob1 : prob1, keep_prob2 : prob2, keep_prob3 : prob3}) 
                    if (val_batches % 20 == 0) or (val_batches == (int(5000/batch_size) - 1)):   
                        vacc = sess.run(accuracy,feed_dict = {x: valid_xx_, y: valid_yy_, keep_prob1 : 1., keep_prob2 : 1., keep_prob3 : 1.})    
                        acc_valid2 += vacc
                        val_co +=1

                    if (val_batches % 50 == 0) or (val_batches == (int(5000/batch_size) - 1)):                        
                        va_err = sess.run(loss ,feed_dict = {x:valid_xx_, y:valid_yy_, keep_prob1 : 1., keep_prob2 : 1., keep_prob3 : 1.})                    
                        err_valid2 +=  va_err
                if (val_batches % 20 == 0) or (val_batches == (int(5000/batch_size) - 1)): 
                    print('Valid Acc:', acc_valid2/batch_size, 'valid err:', err_valid2/batch_size)                   
                acc_valid1 += acc_valid2              
                val_batches += 1              
            valid_acc[k] = acc_valid1/ (val_co)
            
            print('Training Acc  ', train_acc)
            print('Validation Acc  ', valid_acc) 

            prob_test = 1.0
            no_test_images = 5000
            nm_test_batches = int(no_test_images/batch_size)    
            test_accu = np.zeros(nm_test_batches)     
            test_batches = 0            
            for test_batch in get_batches(X_test,  y_test, batch_size, crop=False, distort=False, shuffle=False):
                update_progress(test_batches/int(no_test_images/batch_size)) 
                inputs, targets = test_batch         
                test_acc1=0
                for i in range(batch_size):
                    if Random_noise:
                        inputs[i,:,:,:] = random_noise((inputs[i,:,:,:]),mode='gaussian', var=gaussain_noise_var)
                    test_xx_ = np.expand_dims(inputs[i,:,:,:],axis=0)
                    test_yy_ = np.expand_dims(targets[i,:], axis=0)                           
                    tacc = sess.run(accuracy, feed_dict = {x: test_xx_, y: test_yy_, keep_prob1 : prob_test, keep_prob2 : prob_test, keep_prob3 : prob_test})              
                    test_acc1 += tacc          
                test_accu[test_batches] = test_acc1/ (batch_size)                  
                test_batches += 1   
            print('Test Accuracy', np.amax(test_accu))            
            if (k % 1 == 0):
                con_m1,con_m2,con_m3,con_m4,con_m5,con_m6,fc_m,b_m,conv_s1,conv_s2,conv_s3,conv_s4,conv_s5,conv_s6,fc_s,b_s  = sess.run([weights_mean['m1'] , weights_mean['m2'], weights_mean['m3'], weights_mean['m4'], weights_mean['m5'], weights_mean['m6'], weights_mean['m7'], weights_mean['m8'], sigmas['s1'], sigmas['s2'], sigmas['s3'], sigmas['s4'], sigmas['s5'], sigmas['s6'], sigmas['s7'], sigmas['s8'] ])
                file11 = open(os.path.join(saved_result_path, "BayesCNN_weights_features.pkl"), 'wb')    
                pickle.dump([ con_m1,con_m2,con_m3,con_m4,con_m5,con_m6,fc_m,b_m,conv_s1,conv_s2,conv_s3,conv_s4,conv_s5,conv_s6,fc_s,b_s], file11)
                file11.close()
                save_path = saver.save(sess, os.path.join(saved_result_path, "model"))                 
                 
        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start)          
        fig = plt.figure(figsize=(15,7))
        plt.plot(train_acc, 'b', label='Training acc')
        plt.plot(valid_acc,'r' , label='Validation acc')
        plt.ylim(0, 1.1)
        plt.title("UVI- CNN on CIFAR10 Data")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')        
        plt.savefig(os.path.join(saved_result_path, "EnVI_CNN_on_CIFAR10_Data_acc.png"))       
        plt.close(fig)
       
        print('Training is Completed')
        print('Training Accuracy',np.mean(train_acc))
        print('Validation Accuracy',np.mean(valid_acc))
        print('---------------------')
        # Saving the objects:        
        f = open(os.path.join(saved_result_path, "training_validation_acc.pkl"), 'wb')        
        pickle.dump([train_acc, valid_acc], f)
        f.close()
        
        if Adversarial_noise:
            print('Building Adversarial Noise .....') 
            epoch = 1             
            for k in range(epoch):                
                train_adv_batches = 0
                for minibatch_adv in get_batches(X_train, y_train, batch_size, crop=False, distort=False): 
                    update_progress(train_adv_batches/int(N/batch_size))    
                    inputs, targets = minibatch_adv 
                    y_true_batch = np.zeros_like(targets) 
                    y_true_batch[:, adversary_target_cls] = 1.0                          
                    for i in range(batch_size):
                        xx_ = np.expand_dims(inputs[i,:,:,:],axis=0) 
                        yy_ = np.expand_dims(y_true_batch[i,:], axis=0) 
                        sess.run([optimizer_adversary],feed_dict={x: xx_,y: yy_ ,keep_prob1 : prob1, keep_prob2 : prob2, keep_prob3 : prob3}) 
                        sess.run(x_noise_clip)                     
                    train_adv_batches += 1                
        
        save_path = saver.save(sess, os.path.join(saved_result_path, "model"))    
    else:  
        sess = tf.Session()        
        saver.restore(sess, tf.train.latest_checkpoint(saved_result_path))   
    
    if Random_noise:
        saved_result_path = './EnVI_CIFAR10_with_sigma_{}/epochs_{}/Gaussian_noise_{}/'.format(init_std, epochs,gaussain_noise_std)#
        
    prob_test = 1.0
    no_test_images = 5000
    nm_test_batches = int(no_test_images/batch_size)    
    test_accu = np.zeros(nm_test_batches)
    ref_test_xx =  np.zeros([nm_test_batches, batch_size, image_size, image_size, num_channel])
    ref_test_label = np.zeros([nm_test_batches, batch_size, num_labels])
    cls_true = np.zeros([nm_test_batches, batch_size])
    cls_pred = np.zeros([nm_test_batches, batch_size])
    uncert_ = np.zeros([nm_test_batches, batch_size, num_labels, num_labels])
    sigma_ff = np.zeros([nm_test_batches, batch_size, num_labels, num_labels])
    test_batches = 0
    for test_batch in get_batches(X_test,  y_test, batch_size, crop=False, distort=False, shuffle=False):    
        update_progress(test_batches/int(no_test_images/batch_size)) 
        inputs, targets = test_batch
        ref_test_xx[test_batches,:,:,:,:] , ref_test_label[test_batches,:, : ]= inputs, targets
        cls_true[test_batches, :] =  np.argmax(targets, axis = 1) 
       # cls_pred = np.zeros_like(cls_true)         
        test_acc1=0
        for i in range(batch_size):
            if Random_noise:
                inputs[i,:,:,:] = random_noise((inputs[i,:,:,:]),mode='gaussian', var=gaussain_noise_var)
            xx_test = np.expand_dims(inputs[i,:,:,:],axis=0)
            yy_test = np.expand_dims(targets[i,:], axis=0)             
            cls_pred[test_batches, i], uncert_[test_batches, i, :, :],sigma_ff[test_batches, i, :, :], tacc = sess.run([y_pred_cls, output_sigma,sigma_f, accuracy],feed_dict = {x: xx_test, y: yy_test, keep_prob1 : prob_test, keep_prob2 : prob_test, keep_prob3 : prob_test})              
            test_acc1 += tacc        
        test_accu[test_batches] = test_acc1/ (batch_size)        
        test_batches += 1   
    f2 = open(os.path.join(saved_result_path, "snr_uncert_info.pkl"), 'wb')        
    pickle.dump([uncert_ ,sigma_ff, cls_true, cls_pred], f2)
    f2.close()     
    
  
     
    fig  = plt.figure(figsize=(15,7))
    plt.plot(test_accu, 'r', label='Test acc')
    plt.ylim(0, 1.2)
    plt.title("Ensemble-VI CNN on CIFAR10 Data Test Acc")
    plt.xlabel("Test Batch Number")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')      
    plt.savefig(os.path.join(saved_result_path, "EnVI_CNN_on_CIFAR10_Data_test_acc.png" ))      
    plt.close(fig)   
    
    print('Test Accuracy', np.amax(test_accu))        
    
    
    test_file = open('./test_images_cifar10.pkl', 'rb')
    img_test, test_label =   pickle.load(test_file) 
    test_file.close()
    cls_true =  np.argmax(test_label, axis = 1) 
    cls_pred = np.zeros_like(cls_true)

    uncert = np.zeros([9, num_labels, num_labels]) 
    mean_val = np.zeros([9, 1,num_labels])
    class_score11 = np.zeros([9, 1, num_labels])
    class_score22 = np.zeros([9, 1, num_labels])
    maxmu_out = np.zeros([9, 1,new_size3, new_size3, num_filter[5]])
    maxmuout_g = np.zeros([9,1, new_size3, new_size3, num_filter[5]])
    max_out = np.zeros([9,1, new_size3, new_size3, num_filter[5]])
    maxout_g = np.zeros([9,1, new_size3, new_size3, num_filter[5]])
    sigma_f1 = np.zeros([9, num_labels, num_labels])
    
    for j in range(9):
        if Random_noise:
            img_test[j,:,:,:] = random_noise((img_test[j,:,:,:]),mode='gaussian', var=gaussain_noise_var)
        xx_ = np.expand_dims(img_test[j,:,:,:],axis=0)
        yy_ = np.expand_dims(test_label[j,:], axis=0)
        
        cls_pred[j],mean_val[j,:,:], class_score11[j,:,:], class_score22[j,:,:], uncert[j,:, :], sigma_f1[j,:, :], maxmu_out[j,:,:,:,:],maxmuout_g[j,:,:,:,:], max_out[j,:,:,:,:], maxout_g[j,:,:,:,:] = sess.run([y_pred_cls,prediction,class_score1,class_score2,output_sigma,sigma_f, maxmu,maxmu_g,maxh,maxh_g], feed_dict = {x: xx_, y:yy_, keep_prob1 : prob_test, keep_prob2 : prob_test, keep_prob3 : prob_test})
    images = img_test
    
    f2 = open(os.path.join(saved_result_path, "test_uncert_info.pkl" ), 'wb')    
    pickle.dump([uncert, mean_val, cls_pred, class_score11, class_score22, sigma_f1, maxmu_out, maxmuout_g, max_out, maxout_g], f2)
    f2.close()
    

    if Adversarial_noise:
        adver_example = sess.run(x_noise)        
        file1 = open(os.path.join(saved_result_path, "Bayes_CNN_x_noise.pkl" ), 'wb')       
        pickle.dump( adver_example , file1)
        file1.close()
    
    con_m1,con_m2,con_m3,con_m4,con_m5,con_m6,fc_m,b_m,conv_s1,conv_s2,conv_s3,conv_s4,conv_s5,conv_s6,fc_s,b_s  = sess.run([weights_mean['m1'] , weights_mean['m2'], weights_mean['m3'], weights_mean['m4'], weights_mean['m5'], weights_mean['m6'], weights_mean['m7'], weights_mean['m8'], sigmas['s1'], sigmas['s2'], sigmas['s3'], sigmas['s4'], sigmas['s5'], sigmas['s6'], sigmas['s7'], sigmas['s8'] ])   
    file11 = open(os.path.join(saved_result_path, "BayesCNN_weights_features.pkl" ), 'wb')    
    pickle.dump([ con_m1,con_m2,con_m3,con_m4,con_m5,con_m6,fc_m,b_m,conv_s1,conv_s2,conv_s3,conv_s4,conv_s5,conv_s6,fc_s,b_s], file11)
    file11.close()
    
    if Adversarial_noise:  
        noise = sess.run(x_noise)        
        print("Noise:")
        print("- Min:", noise.min())
        print("- Max:", noise.max())       
        plt.axis('off')                 
        plt.imsave(os.path.join(saved_result_path, "EnVI_on_CIFAR10_noise.png"), toimage(noise), cmap='seismic', vmin=-1.0, vmax=1.0) 
         
        # Plot the first 9 images.
        plot_images(images=images[0:9,:,:,:],sigma_std=init_std,epoch=epochs,cls_true=cls_true[0:9],path = saved_result_path, cls_pred=cls_pred[0:9],  noise=noise)
    else:
        plot_images(images=images[0:9,:,:,:],sigma_std=init_std,epoch=epochs,cls_true=cls_true[0:9],path=saved_result_path, cls_pred=cls_pred[0:9]) 
             
    textfile = open(os.path.join(saved_result_path, "Related_info.txt"),'w')    
    textfile.write(' Number of kernels : ' +str(num_filter)) 
    textfile.write("\n---------------------------------")    
    textfile.write("\n Test Accuracy : "+ str(np.amax(test_accu)))   
    if Training:
        textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc)))
        textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc)))
        textfile.write('\n Total run Time in sec : ' +str(stop - start))
    textfile.write("\n---------------------------------")
    if Random_noise:
        textfile.write('\n Random Noise std: '+ str(gaussain_noise_std))       
        textfile.write("\n---------------------------------")
    if Adversarial_noise:
        textfile.write('\n Noise limit: '+ str(noise_limit))
        textfile.write("\n- Min : "+ str(noise.min())) 
        textfile.write("\n- Max : "+ str(noise.max()))         
        textfile.write("\n---------------------------------")      
        textfile.write("\n Adversary LR : "+ str(adv_lr))            
            
    textfile.write("\n---------------------------------")    
    textfile.write('\n Initial std of sigma of the weights : ' +str(init_sigma_std))    
    textfile.write('\n Initial log(1+exp(sigma)) : ' +str(init_std)) 
    textfile.write('\n Rate of Convergence : ' + str(epochs)+ ' epochs')  
    textfile.write('\n Initial STD of the mean of the weights : ' + str(weight_std))  
    textfile.close()
    sess.close()   
        
if __name__ == '__main__':
    main_function()    
