import numpy as np
 
def sgm(x,deriv=False):

    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0.05, 0.1, 1]])

y = np.array([[0.99, 0.01]])

syn0 = np.array([[0.14, 0.2],
                [0.25, 0.32],
                [0.3, 0.3]])
syn1 = np.array([[0.42, 0.42],
                 [0.38, 0.5],
                 [0.55, 0.55]])

r = 130000
for j in range(r):
    
    l0 = X
    l1 = sgm(np.dot(l0,syn0))
  
    h1 = l1[0][0]
    h2 = l1[0][1] 
    l1_b = np.array([[h1,h2,1]])
   
    l2 = sgm(np.dot(l1_b,syn1))


    l2_error = y - l2

    error = 0.5*np.sum((l2_error)**2)
  
    if (j%100) == 0:
        print ("Error: {0} , Iteration: {1} ".format(error,j))

    #update 
    learningRate = 0.08
    l2_delta = -l2_error*sgm(l2,deriv=True)       
        
    l1_error = l2_delta.dot(syn1.T) 
    
    l1_delta = l1_error * sgm(l1_b,deriv=True)

    syn0[0][0] -= l1_delta[0][0]*X[0][0]*learningRate
    syn0[0][1] -= l1_delta[0][1]*X[0][1]*learningRate
    syn0[1][0] -= l1_delta[0][0]*X[0][0]*learningRate
    syn0[1][1] -= l1_delta[0][1]*X[0][1]*learningRate
    syn0[2][0] -= l1_delta[0][0]*X[0][2]*learningRate
    syn0[2][1] -= l1_delta[0][1]*X[0][2]*learningRate

    syn1[0][0] -= l2_delta[0][0]*h1*learningRate
    syn1[0][1] -= l2_delta[0][1]*h2*learningRate
    syn1[1][0] -= l2_delta[0][0]*h1*learningRate
    syn1[1][1] -= l2_delta[0][1]*h2*learningRate
    syn1[2][0] -= l2_delta[0][0]*learningRate
    syn1[2][1] -= l2_delta[0][1]*learningRate
    

