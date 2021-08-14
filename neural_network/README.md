## Single Layer Neural Network

This directory contains code for a single-layer neural network that I wrote as part of my coursework in ML-10601 at Carnegie Mellon. Along with the code are  training and testing data for testing the code in the form of a small and large dataset for classifying images of handwritten letters. This implementation opitmizes the parameters of the single-layer neural network using stochastic gradient descent (SGD). 

### Running neuralnet.py

neuralnet.py can be tested by running a command line argument like the following

```
python neuralnet.py training_data.csv testing_data.csv train_out.labels test_out.labels metrics_out.txt num_epochs hidden_units init_flag learning_rate
```

training_data.csv and testing_data.csv contain the training and testing data, respectively. train_out.labels and test_out.labels will contain the neural network's prediction on the training and test data. metrics_out.txt will contain the training and test error. Specify the number of epochs using num_epochs, the number of hidden nodes in the hidden layer with hidden_units, how you would like to initialize the weights using init_flag (RANDOM = 1, ZERO = 2), and the learning rate with learning_rate.

To test out neuralnet.py on the large dataset provided using a neural network with 50 hidden nodes, 100 epochs, random weight initialization, and a learning rate of 0.001, run the following command in your terminal from the neural_network directory

```
python neuralnet.py largeTrain.csv largeTest.csv large_train_out.labels large_test_out.labels large_metrics_out.txt 100 50 1 0.001
```

Note that if you are currently a student at CMU and are seeking a solution for a homework assignment, you are encouraged to cease looking through this repository and brave the challenges of the course on your own (or ethically with some of your classmates :) ). 