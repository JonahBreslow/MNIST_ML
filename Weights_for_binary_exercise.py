import numpy as np

'''
Defining Sigmoid "Squishification Function"
'''
def sigmoid(x):
    return(1/(1+np.exp(-x)))

'''
Defining the w.x + b for each output neuron
# note that w_0 is the vector of weights used to calculate the output
node a_0, etc.
'''
def new_representation(activation_vector):
    a_0 = np.sum(w_0 * activation_vector)
    a_1 = np.sum(w_1 * activation_vector)
    a_2 = np.sum(w_2 * activation_vector)
    a_3 = np.sum(w_3 * activation_vector)

    return a_3, a_2, a_1, a_0

'''
Defining the function that applies the sigmoid "Squishification" function
to the w.x + b vectors
'''
def new_repr_binary_vec(new_representation_vec):
    sigmoid_op = np.apply_along_axis(sigmoid, 0, new_representation_vec)
    return (sigmoid_op > 0.99).astype(int)

'''
Creating the w vectors
    1) any number that has the first bit is in w_0
        (1,3,5,7,9) --> (1000, 1100, 1010, 1110, 1001)
    2) any number that has second bit is in w_1
        (2,3,6,7)  --> (0100, 1100, 0110, 1110)
    etc.

Arbitrarily make weights 10 when the respective bit is in the number, and
-10 when the bit is not in the number
        '''
w_0 = np.full(10, -10, dtype=np.int8)
w_0[[1, 3, 5, 7, 9]] = 10
w_1 = np.full(10, -10, dtype=np.int8)
w_1[[2, 3, 6, 7]] = 10
w_2 = np.full(10, -10, dtype=np.int8)
w_2[[4, 5, 6, 7]] = 10
w_3 = np.full(10, -10, dtype=np.int8)
w_3[[8, 9]] = 10

'''
Creating activation vector. When [3] is changed to 0.99, that means that in the
penultimate neuron layer, the network recognized the output value as 2 (index 3)
'''
activation_vec = np.full(10, 0.01, dtype=np.float)
# print(activation_vec)
activation_vec[3] = 0.99
# print(activation_vec)

'''
Finally, call the new_representation Function
then pass the outputs through the "Squishification" Function
and show only outputs that are "activated" (which we define as being >.99 above)
'''
new_representation_vec = new_representation(activation_vec)
print("New Representation Vector")
print(new_representation_vec)

print(new_repr_binary_vec(new_representation_vec))


# if you wish to convert binary vector to int
b = new_repr_binary_vec(new_representation_vec)
print(b.dot(2**np.arange(b.size)[::-1]))
