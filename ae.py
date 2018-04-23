# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#we are using stacked auto encoders which are nothing but auto encoders with multiple hidden layers


# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        
        # first full connection ie. fully connected layer of input nodes to first hidden layer
        #we are using nn.Linear func for this which accepts 2 arguments ist one is no. of input nodes
        #and second one is no. of nodes in the hidden layer
        #here no. of nodes in the hidden layers is completely expt based

        
        self.fc1 = nn.Linear(nb_movies, 20)  
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)  #start decoding the features(10 in input and 20 in output)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid() #defining the activation func to activate the neurons in these 4 fully connected layers

     #defining the func that will define the acton of encoding and decoding 
     #and also applying the diff activation func in fc layers
     #main purpose is to return the vector of predicting ratings
     
    def forward(self, x): #here x is the input vector
        ''''
        first encoding takes place that is encoding the input vector to a shorter vector of 20 nodes
        for encoding to take place we have to apply the activation func to the whole fc layer
        ie. self.activation(self.fc1(x))  will return the ist encoded vector
        and since we have to use this encoded vector further in the process hence lets assign it to x.
        '''

        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        # final decoding to get the output vector,but this time we won't use activation func
        #but rather we use simply self.fc4(x--) > decoded vector from prev. step
        x = self.fc4(x)
        #finally returning the vector of predicting ratings
        return x
sae = SAE()
#defining the criterion for the loss func and loss func will be the mean sqare error
criterion = nn.MSELoss()
#line 62: now we have to define an optimizer that will apply the stochastic gradient descent
#to update the weights so as to minimize the error

# we are using the RMSprop as the optimizer which will also accept 3 arguments
#ist one is all the parmaters that is needed to define the architecture of our auto-encoder
#to get all the parameters of our auto encoder we will use sae.parmeters() where sae is the obj of our encoder
#now the second parameter is lr ie. learning rate totally expt  based 
#3rd parmater is decay which is used to reduce the learning rate in every epoch and that inorder to regulate the convergence

optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
# now we iterate over all the observations over all the epochs
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 #initilaize the loss error as train loss to 0.
    #variable that will keep track of the no. of users that rated at least 1 movie
    #this is to make sure that we don't include the users who don't rate any
    # we will initialize this variable as float ==0. bcs we will use it to compute the root mean square error

    s = 0.
    
    # start the 2nd loop to iterate over all the obs in a epoch , in this epoch we will do all our actions , perform encoding,decoding etc
    
    for id_user in range(nb_users):
        
        # input is the input vector of all the ratings given by this given user
        #but our func will not accept this single vector of inputs, what we have to do is to create an additional dimenson of batch
        #so basically we have to create a batch of input vector otherwise it won't work

        
        input = Variable(training_set[id_user]).unsqueeze(0)
        #now initilaize an other variable target which is just a copy of original input vector

        target = input.clone()
        #condition to make sure to look only the users who rated at least a movie
        #so if an obs contains only 0's then we will not consider that obs
        #target.data is just all the ratings from target(basically no. of rating>0)
        if torch.sum(target.data > 0) > 0:
            
            #set output = vector of our predicted ratings, we have to call our forward func for this on variable input
            #we can do this by sae(input)
            output = sae(input)
            
            #for optimizing purpose we set target.require_grad = false it means that we want to
            #calculate the gradient wrt to input and not target, this will reduce a huge reduntant calculations
            
            target.require_grad = False
            #next step is also for optmizing we set output[target==0] =0 ie. we are settting all the values corresponding to the index of value=0 in target to 0
            #so that we only deal with non-zeroes entries

            output[target == 0] = 0
            #computation of loss using our criterion obj which will take 2 arguments  vector of predicting rating and actual rating  
            loss = criterion(output, target)
            #we are defining an other variable mean_corrector(representing avg of err but by only considering the movies that gets at least 1 rating)= no. of movies/no. of movies which is +vely rated
            #we are adding some small +ve qty in denominator so that it can never be 0
            #torch.sum(target.data>0) count all the positive entries in the target
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            
            # call loss.backward just to find the direction in which weights must be updated

            
            loss.backward()
            #loss.data[0] will represent the square error between actual and predicted
            #now we are multiplying it to the mean_corrector so as to get the relevant loss
            #and since it is sqaure err diff we have to take sqrt of this
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            #finally use the optimizer to actually update the weights
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
    
'''    
training set contains actual rating given by the user, but it also contain some movies
which is not watched by user so baically we are training our model ro predict the rating for the movies which is not watched by user
and then finally our test set contains the actual rating for those movies
so we compare our result to the test set rating
''' 
    
    
    
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)#we dont need to change it to test set bcos we are predicting for the movies not watched by user 
    target = Variable(test_set[id_user]) #and test set contains the actual future rating
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))