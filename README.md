# SISE2024-taks2 - UWB Localization Correction with Neural Networks

## Project Overview
This repository contains the implementation of a multilayer perceptron neural network designed to correct inaccuracies in the localization of robots within buildings using Ultra-Wideband (UWB) technology. The project is developed in Python and involves creating and training a neural network from scratch.

## Structure of the Neural Network
- **Input Layer**: Consists of 2 neurons to receive input signals from UWB sensors.
- **Hidden Layers**: One or more nonlinear hidden layers, where the number of neurons and layers can be easily adjusted.
- **Output Layer**: A linear output layer with 2 neurons outputting the corrected coordinates of the robot.

## Features 
- **Customizable Architecture**: Users can modify the number of hidden layers and neurons per layer.
- **Activation Functions**: Implementation includes ReLU, sigmoid, and tanh activation functions.
- **Learning Parameters**: Adjustable learning rate, along with options to configure other learning parameters such as momentum and weight decay.

## Implementation Details
- **Backpropagation Algorithm**: Used for training the neural network to minimize the mean squared error between the predicted and actual robot positions.
- **Loss Function**: Utilizes Mean Squared Error (MSE) to evaluate the performance of the network.
- **Optimization Algorithms**: Supports various optimizers like SGD, Adam, and RMSprop to enhance the learning process.
