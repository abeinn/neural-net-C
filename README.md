# Description
Implementation of a neural network framework in plain C. 
# How It Works
Matrix (and other math) operations are defined in `math_utils.c`. Neural network operations (create, train, evaluate) are defined in `neural_network.c`. Currently, the ReLu, Sigmoid, and Softmax activation functions are supported, along with multiple layers and varying nodes in each layer. Training works by performing forward prop, back prop, and a gradient descent update. After training finishes, the training accuracy is computed on the entire training set. 

Users can use the functions defined in `neural_network.c` to create and train a model given input data and labels (in the form of a .csv file), as shown in `xor_model.c` and `mnist_model.c`. They can then evaluate the model on test data using the `model_predict` function, and output the predictions to a .csv file. 

