"""
ERICH KRAMER 
"""

import sys
import pdb
import pickle
import numpy as np
from operator import add

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    #take in W as dimensions needed
    def __init__(self, W):
        print(W)
        #W = (W[0]+1, W[1])
        self.W = np.random.rand( *W )*.2 -.1 #here the inputs are already scaled for x bias
        self.bias = np.ones(( 1, W[1]) )*.1

        self.v = 0
        

    def forward(self, x):
        return np.dot( x , self.W) + self.bias   #W is fromXto, x is input 1Xfrom

    def backward( self, grad_output, zin,
            learning_rate=  0.1, momentum=0.1, l2_penalty=0.0 ):
        delt = np.dot( np.matrix(zin).T , np.matrix(grad_output) )
        self.v = self.v*momentum +delt*learning_rate #learning_rate*grad_output
        self.W -= self.v
        self.bias -= grad_output*learning_rate
        print(self.W)
        return delt#zin * grad_output


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        return x * ( x > 0 )  #equivalent to x if x >0 ; else 0
        # DEFINE forward function

    #on backward pass we go from 
    def backward( self, grad_output, zin,
            learning_rate=0.1, momentum=0.1, l2_penalty=0.1 ):
        return np.multiply( grad_output.T, 1 * (zin > 0 ) ) #filter those < 0, map to 1
        
        #otherwise
        #for z in zin:
        #    if z > 0:

        #if zin > 0:
        #    return grad_output #*1 but waste to include
        #return 0
    # DEFINE backward function


# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class Sigmoid(object):
    #target is the real value 
        def forward(self, x):
            x = np.clip(x, -10, 10)
            tmp = 1.0/(1.0+np.exp(-x) )
            return tmp
    
        def backward( self, grad_output, zin, *args, **kwargs):
            zin = np.clip(zin, -10, 10) 
            tmp = np.exp(-zin)/( (np.exp(-zin)  +1)**2)
            return tmp*grad_output


        #predict class 0 or 1 based on sigmoid mapping from assignment desc.
        def predict( self, x):
            #x = np.clip(x, -10, 10)
            if x > .5:#(1/(1+np.exp(-x) )) > .5:
                return 1
            else:
                return 0
            
class CrossEntropy(object):
    def forward(self, x, target):
        x = np.clip(x, .001, .999)
        return -( target*np.log(x) + (1.0-target)*np.log(1.0-x))
    

    def forward2(self, x, target):
        x = np.clip(x, .001, .999)
        sig = 1.0/(1+np.exp(-x))
        return - (target*np.log(sig) + (1-target)*np.log(1 - sig) )

    #to calculate this we need forward result
    def backward(self, grad_output, target):

        return grad_output - target


    def backward2(self, grad_output, target):
        grad_output = np.clip(grad_output, .001, .999)
        tmp = - ( target/grad_output + (1-target) / (1-grad_output))
        return tmp


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input = input_dims
        self.nhidden = hidden_units
        
        self.layers = []
        W1_dim = (input_dims, hidden_units)
        self.layers.append( LinearTransform( W1_dim ) )
        self.layers.append( ReLU() )
        W2_dim = (hidden_units, 1)
        self.layers.append( LinearTransform( W2_dim ) )
        self.layers.append( Sigmoid() )
        self.layers.append( CrossEntropy() )
    #w1 input_dimsXhidden_units
    #ReLu hiddenXhidden
    #w2
    #sigmoid
    #loss


    def train( self, x_batch, y_batch, 
        lr=.5, mtm=.1, l2_p=.1 ):      #train on variable sizebatch
        loss = 0

        pipeSum = [0]*4
        pipe = [0]*4 
        #pipe = out1, g(out1), out2, sig(out2), cEnt(sig(out2)
        batch_size = float(len(x_batch))
        xsum = np.zeros(x_batch[0].shape)

        for x,y in zip(x_batch, y_batch):
            #FORWARD AND STORE
            pipe = []
            pipe.append( self.layers[0].forward(x) )
            for i,lyr in enumerate(self.layers[1:-1]):
                pipe.append(lyr.forward(pipe[i]) )
            #pipe( self.layers[-1].forward(pipe[-1], y) )


            pipeSum = map(add, pipeSum, pipe)
            xsum += x

        #BACKWARDS

        xsum = sum(x_batch)/ float(batch_size)
        #pipe = [p/float(batch_size) for p in pipeSum]
        pipe = [ p/batch_size for p in pipeSum]
        siphon = []
        siphon.append( self.layers[-1].backward( pipe[-1], y) )
        for lyr, p in zip( self.layers[::-1][1:], pipe[::-1][1:] ):
            siphon.append( lyr.backward(siphon[-1], p, lr, mtm, l2_p) )
        siphon.append( self.layers[0].backward( siphon[-1], xsum, lr, mtm, l2_p ) )

        

        pdb.set_trace()

        
        return  #loss, accuracy #RETURN ACCURACY AND LOSS FOR THIS SUBSET



    def evaluate(self, x_batch, y_batch):

        loss = correct = total = 0

        for x,y in zip(x_batch, y_batch):
            pipe = []
            pipe.append(self.layers[0].forward(x) )
            for lyr in self.layers[1:-1]:
                pipe.append(lyr.forward( pipe[-1] ) )
            pred = self.layers[-2].predict( pipe[-1] )

            if pred == y:
                correct +=1
            total +=1
            loss += self.layers[-1].forward(pipe[-1], y )

        accuracy = int(correct/total * 100)

        return  loss, accuracy
        # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    data = pickle.load(open('cifar-2class-py2/cifar_2class_py2.p3', 'rb'))

    #pdb.set_trace()

    train_x = data[b'train_data']/127.5-1#normalized to [-1,1]; 
    train_y = data[b'train_labels']
    test_x = data[b'test_data']/127.5-1  #normalized to [-1,1]; 
    test_y = data[b'test_labels']

    num_examples, input_dims = train_x.shape
     
    hidden_units = 1000 #input_dims * 2
    num_epochs = 10
    num_batches = 100
    batch_size = float(len(train_x))/float(num_batches)


    lr = .1
    momentum = .1
    l2_penalty = .1

    mlp = MLP(input_dims, hidden_units)

    test_loss = test_accuracy = 0
    for epoch in range(num_epochs):

        # INSERT YOUR CODE FOR EACH EPOCH HERE
        
        total_loss = 0
        rng = np.random.get_state()
        np.random.shuffle(train_y)
        np.random.set_state(rng)
        np.random.shuffle(train_x)

        for b in range(num_batches):
            start = float(b) * batch_size
            end = start + batch_size
            cut = slice(int(start), int(end))

            print("\t", cut)


            #TRAIN
            mlp.train( train_x[cut], train_y[cut], lr, momentum, l2_penalty)
            
            train_loss, train_accuracy = mlp.evaluate( train_x[cut], train_y[cut] )

            #REFRESH TEST RESULTS, 
            #test_loss, test_accuracy = mlp.evaluate( test_x, test_y )

            total_loss += train_loss

            print('\r[Epoch {}, mb {}]  Avg.Loss = {}, Accuracy={}%'.format(
                    epoch + 1, b +  1, total_loss/(b+1), train_accuracy ), end='', )

            sys.stdout.flush()
                # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
                # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    

        train_loss, train_accuracy = mlp.evaluate( train_x, train_y)
        print('\n    Train Loss: {}    Train Acc.: {}%'.format(
            train_loss, train_accuracy) )

        test_loss, test_accuracy = mlp.evaluate( test_x, test_y )
        
        print('    Test Loss:  {}    Test Acc.:  {}%'.format(
            test_loss,  test_accuracy ) ) 





