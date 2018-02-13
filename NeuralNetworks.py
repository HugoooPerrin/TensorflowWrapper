#!/usr/bin/env python
# -*- coding: utf-8 -*- 


#---------------------------------------------------------------------------------------------------------------------
"""
The present algorithm is based on the Google Tensorflow architecture.


Last update : 24/09/2017
Author : Hugo Perrin

"""



#========================================================================================================================================
# LIBRAIRIES
#========================================================================================================================================




import tensorflow as tf
import numpy as np
from math import floor
from math import sqrt
from datetime import datetime
from dateutil.relativedelta import relativedelta




#========================================================================================================================================
# UTILS
#========================================================================================================================================




def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)





def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])




# TO REPLACE BY sklearn.cross_validation.train_test_split

def random_sample(data, target, sampling_rate):
    array = np.arange(data.shape[0])
    sample = floor(data.shape[0]*sampling_rate)
    np.random.shuffle(array)
    sample_mask = array[:sample]
    train_mask = array[(sample+1):]
    Xsample = data[sample_mask,:]
    Ysample = target[sample_mask,:]
    Xtrain = data[train_mask,:]
    Ytrain = target[train_mask,:]

    return Xtrain, Ytrain, Xsample, Ysample





#========================================================================================================================================
# PERCEPTRON
#========================================================================================================================================





def multilayer_perceptron(data, weights, biases, dropout = None, keep_prob = None):

    layers = {}

    # Hidden layer with RELU activation

    layers["hidden0"] = tf.add(tf.matmul(data, weights["hidden0"]), biases["hidden0"])
    layers["hidden0"] = tf.nn.relu(layers["hidden0"])


    # Dropout on hidden layer

    if dropout is not None:
        layers["hidden0"] = tf.nn.dropout(layers["hidden0"], keep_prob)


    for i in range(len(weights)-2):

        # Hidden layer with RELU activation

        layers["hidden{}".format(i+1)] = tf.add(tf.matmul(layers["hidden{}".format(i)], weights["hidden{}".format(i+1)]), biases["hidden{}".format(i+1)])
        layers["hidden{}".format(i+1)] = tf.nn.relu(layers["hidden{}".format(i+1)])


        # Dropout on hidden layer

        if dropout is not None:
            layers["hidden{}".format(i+1)] = tf.nn.dropout(layers["hidden{}".format(i+1)], keep_prob)
        

    # Output layer with linear activation
    out_layer = tf.matmul(layers["hidden{}".format(len(weights)-2)], weights['hidden{}'.format(len(weights)-1)]) + biases['hidden{}'.format(len(weights)-1)]
    return out_layer





#========================================================================================================================================
# NEURAL NETWORK CLASSIFIER
#========================================================================================================================================




class NeuralNetworkClassifier(object):




    def __init__(self, layers, num_steps, display_step, learning_rate, L2Regression = None, dropout = None, 
                 learning_rate_decay = None, batch_size = None, verbose = None):



        # Layers

        self.layers = layers
        self.dept = len(layers)


        # Regularization

        if L2Regression is not None:
            self.beta = L2Regression
        else:
            self.beta = None
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = None


        # Optimizer algorithm 

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = None


        # Learning parameters

        if learning_rate_decay is not None:
            self.learning_rate_decay = learning_rate_decay
        else:
            self.learning_rate_decay = None
        self.learning_rate = learning_rate


        self.num_steps = num_steps
        self.display_step = display_step


        self.summary(L2Regression, dropout, learning_rate_decay, batch_size, verbose)





    def summary(self, L2Regression, dropout, learning_rate_decay, batch_size, verbose):



        if verbose is not None:
            print("-------------------- PROCESSING INITIALIZATION --------------------")
            print("\nModel : \n")
            for layer in range(len(self.layers)):
                print(" -> Hidden layer n°{} contains {} nodes".format(layer+1,self.layers[layer]))
            print("\nMethod : \n")
            if batch_size is not None:
                print(" -> Stochastic Gradient Descent (Batch size : {})".format(self.batch_size))
            else:
                print(" -> Batch Gradient Descent")
            print("\nParameters : \n")
            print(" -> Number of steps : {}".format(self.num_steps))
            if learning_rate_decay is not None:
                print(" -> Learning rate : {} with exponential learning rate decay ({})".format(self.learning_rate,self.learning_rate_decay))
            else: 
                print(" -> Learning rate : {} with no learning rate decay".format(self.learning_rate))
            print("\nRegularization : \n")
            if L2Regression is not None:
                print(" -> L2 Regression beta : {}".format(self.beta))
            else:
                print(" -> No L2 Regression")
            if dropout is not None:
                print(" -> dropout probability : {}".format(self.dropout))
            else: 
                print(" -> No dropout")
            print(" \n\n     >> NEURAL NETWORK INITIALIZED")





    def fit(self, Xtrain, Ytrain, Xtest = None, Ytest = None, validation = None):

        """
            The target datasets shall have individuals as rows and classes as
            columns ! 

            The feature datasets shall have individuals as rows and features as
            columns !  

            All value must be float32
        """
        
        print("\n-------------------- PROCESSING LEARNING --------------------\n")

        time1 = datetime.now()

        #======================================================================================================
        # NEURAL NETWORK CONSTRUCTION : 
        #======================================================================================================


        #===============================================
        # VARIABLES : 
        #=============================================== 

        num_features = Xtrain.shape[1]
        num_labels = Ytrain.shape[1]


        #===============================================
        # VALIDATION - TRAIN SPLIT : 
        #===============================================

        if validation is not None:
            Xtrain, Ytrain, Xvalid, Yvalid = random_sample(Xtrain, Ytrain, validation)


        #===============================================
        # GRAPH DEFINITION : 
        #===============================================

        graph = tf.Graph()
        with graph.as_default():


            #===========================================
            # TRAINING DATA : 
            #===========================================

            if self.batch_size is not None:

                # Use of a placeholder that will be fed at run time with a training minibatch.
                tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, num_features))
                tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))

            else:

                #Use of all the training dataset
                tf_train_dataset = tf.constant(Xtrain)
                tf_train_labels = tf.constant(Ytrain)


            #===========================================
            # VALIDATION DATA : 
            #===========================================

            if validation is not None:
                    tf_valid_dataset = tf.constant(Xvalid)


            #===========================================
            # TEST DATA : 
            #===========================================

            if Xtest is not None:
                tf_test_dataset = tf.constant(Xtest)


            #===========================================
            # DROPOUT : 
            #===========================================

            if self.dropout is not None:
                keep_prob = tf.placeholder(tf.float32)


            #===========================================
            # WEIGHTS : 
            #===========================================

            self.weights = {}

            self.weights["hidden0"] = tf.Variable(tf.truncated_normal([num_features, self.layers[0]], stddev = sqrt(2.0/(num_features))))

            for i in range(self.dept-1):
                self.weights["hidden{}".format(i+1)] = tf.Variable(tf.truncated_normal([self.layers[i], self.layers[i+1]], stddev = sqrt(2.0/(self.layers[i]))))

            self.weights["hidden{}".format(self.dept)] = tf.Variable(tf.truncated_normal([self.layers[self.dept-1], num_labels], stddev = sqrt(2.0/(self.layers[self.dept-1]))))


            #===========================================
            # BIASES :
            #===========================================

            self.biases = {}

            self.biases["hidden0"] = tf.Variable(tf.zeros([self.layers[0]]))

            for i in range(self.dept-1):
                self.biases["hidden{}".format(i+1)] = tf.Variable(tf.zeros([self.layers[i+1]]))

            self.biases["hidden{}".format(self.dept)] = tf.Variable(tf.zeros([num_labels]))


            #===========================================
            # PERCEPTRON :
            #===========================================

            if self.dropout is not None:
                training_model = multilayer_perceptron(tf_train_dataset, self.weights, self.biases, self.dropout, keep_prob)
            else:
                training_model = multilayer_perceptron(tf_train_dataset, self.weights, self.biases, self.dropout)


            #===========================================
            # LOSS FUNCTION : 
            #===========================================

            # Original loss function :
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = training_model))


            #===========================================
            # L2 REGULARIZATION : 
            #===========================================

            if self.beta is not None:

                regularizer = 0

                for i in range(len(self.weights)):
                    regularizer += tf.nn.l2_loss(self.weights["hidden{}".format(i)])

                loss = tf.reduce_mean(loss + self.beta * regularizer)


            #===========================================
            # LEARNING RATE DECAY : 
            #===========================================

            if self.learning_rate_decay is not None:

                # count the number of steps taken.
                global_step = tf.Variable(0)

                self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, self.learning_rate_decay, staircase=True)


            #===========================================
            # OPTIMIZER ALGORITHM : 
            #===========================================

            if self.learning_rate_decay is not None:

                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step = global_step)

            else:

                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)


            #===========================================
            # ACCURACY : 
            #===========================================

            # NO DROPOUT ON PREDICTION, ONLY DURING TRAINING

            # TRAIN 
            train_prediction = tf.nn.softmax(multilayer_perceptron(tf_train_dataset, self.weights, self.biases, dropout = None))

            # VALIDATION
            if validation is not None:                                                      
                valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset, self.weights, self.biases, dropout = None)) 

            # TEST
            if Xtest is not None:
                test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, self.weights, self.biases, dropout = None))



        #======================================================================================================
        # NEURAL NETWORK OPTIMIZATION : 
        #======================================================================================================

        with tf.Session(graph=graph) as session:

            #===================================================
            # VARIABLES INITIALIZATION : 
            #===================================================

            tf.global_variables_initializer().run()


            #===================================================
            # STEPS : 
            #===================================================

            for step in range(self.num_steps):


                #===============================================
                # MINI-BATCH : 
                #===============================================

                if self.batch_size is not None:

                    # Pick an offset within the training data, which has been randomized
                    # Note: we could use better randomization across epochs

                    offset = (step * self.batch_size) % (Ytrain.shape[0] - self.batch_size)

                    # Generate a minibatch

                    batch_data = Xtrain[offset:(offset + self.batch_size), :]
                    batch_labels = Ytrain[offset:(offset + self.batch_size), :]


                    #===========================================
                    # OPTIMIZATION : 
                    #===========================================

                    # Prepare a dictionary telling the session where to feed the minibatch.
                    # The key of the dictionary is the placeholder node of the graph to be fed,
                    # and the value is the numpy array to feed to it.

                    if self.dropout is not None:

                        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : self.dropout}
                        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

                    else:

                        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)


                #===============================================
                # FULL BATCH : 
                #===============================================

                else:

                    if self.dropout is not None:

                        feed_dict = {keep_prob : self.dropout}
                        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

                    else:
                        _, l, predictions = session.run([optimizer, loss, train_prediction])


                #===============================================
                # LEARNING ACCURACY EVALUATION : 
                #===============================================

                if (step % self.display_step == 0):

                    if self.batch_size is not None:

                        print("Step : {}   Minibatch loss : {}   Validation accuracy: {:.1f}".format(step, l, accuracy(valid_prediction.eval(), Yvalid)))

                    else :

                        print("Step : {}   Loss : {}   Validation accuracy: {:.1f}".format(step, l, accuracy(valid_prediction.eval(), Yvalid)))

                 

            if Xtest is not None:            
                print("\n>> Test accuracy: {:.1f}\n".format(accuracy(test_prediction.eval(), Ytest)))


            #===================================================
            # TIME EVALUATION : 
            #===================================================

            print("Optimization time : {}".format(diff(datetime.now(),time1)))





    def predict(self, features):
        """
            The predictions should be COMPUTED WIHTOUT DROPOUT.
            The feature datasets shall have individuals as rows and features as columns !  
            All value must be float32.
        """
