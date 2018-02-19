"""
ERICH KRAMER 
"""


from __future__ import division
from __future__ import print_function

import sys
import pdb
import cPickle
import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W):
        self.W = np.concatenate( ( W, np.ones((1,W.shape[1])) ), axis=0 )
        #append a bias dimension so that everything is less awful. 
        #np.concat is very slow, do this outside the function for faster performance
        

    def forward(self, x):
        return np.dot( x , self.W)   #W is fromXto, x is input 1Xfrom

    def backward( self, grad_output, 
        learning_rate=0.0, momentum=0.0, l2_penalty=0.0 ):
        
        return None         #TBI


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        return x * ( x > 0 )  #equivalent to x if x >0 ; else 0
        # DEFINE forward function

    def backward( self, grad_output, 
        learning_rate=0.0, momentum=0.0, l2_penalty=0.0 ):
        if x > 0:
            return 1
        return 0
    # DEFINE backward function


# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class Sigmoid(object):
    #target is the real value 
        def forward(self, x , target):
            return  1/(1+np.exp(x) )

            
        def backward( self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
            #asdfasdfasf
            return 0


        #predict class 0 or 1 based on sigmoid mapping from assignment desc.
        def predict( self, x):
            if 1/(1+np.exp(x) ) > .5:
                return 1
            else:
                return 0
            
class CrossEntropy(object):
    def forward(self, x, target):
        
        return

    def backward(self, x, target):
        return





# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input = input_dims
        self.nhidden = hidden_units
        
        self.layers = []
    #w1
    #w2


    def train( self, x_batch, y_batch, 
        learning_rate=.5, momentum=.1, l2_penalty=.1 ):      #train on variable sizebatch
        # INSERT CODE for training the network
        
        
        #RETURN ACCURACY AND LOSS FOR THIS SUBSET
        accuracy = loss = 0

        return  loss, accuracy
    def evaluate(self, x_batch, y_batch):
        

        accuracy = loss = 0
        return  loss, accuracy
        # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    data = cPickle.load(open('cifar-2class-py2/cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']/255#normalized to [0,1]; consider [-1,1]
    train_y = data['train_labels']
    test_x = data['test_data']/255  #normalized to [0,1]; consider [-1,1]
    test_y = data['test_labels']
        
    num_examples, input_dims = train_x.shape
     
    hidden_units = input_dims  
    num_epochs = 10
    num_batches = 100
    batch_size = float(len(train_x))/float(num_batches)


    lr = .5
    momentum = .1
    l2_penalty = .1

    mlp = MLP(input_dims, hidden_units)


    total_loss = 0
    for epoch in xrange(num_epochs):

        # INSERT YOUR CODE FOR EACH EPOCH HERE
        
        for b in xrange(num_batches):
            start = float(b) * batch_size
            end = start + batch_size
            cut = slice(int(start), int(end))

            print(cut)
            pdb.set_trace()


            #TRAIN
            train_loss, train_accuracy = mlp.train( train_x[cut], train_y[cut], 
                    lr, momentum, l2_penalty)
            
            #REFRESH TEST RESULTS, 
            test_loss, test_accuracy = mlp.evaluate( test_x, test_y )

            total_loss += test_loss
                        # MAKE SURE TO UPDATE total_loss


            print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1, b +  1, total_loss, ), end='', )
            sys.stdout.flush()
                # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
                # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy


        train_loss, train_accuracy = mlp.train( train_x[cut], train_y[cut], 
                lr, momentum, l2_penalty)

        test_loss, test_accuracy = mlp.evaluate( test_x, test_y )
        
        print('\n    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss, 100. * train_accuracy) )
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss, 100. * test_accuracy ) ) 





