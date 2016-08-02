##Synopsis

A Python-based Neural Network library. Currently, a single hidden layer feedforward network is implemented. Eventually, multiple layers, and other network types including Restricted Boltzmann Machines, Recurrent Neural Networks (RNN), and Convolution Neural Networks (CNN) will be added.

##Quick start:

To run the neural network classifier on the example EEG_Eye dataset, type the following command at the prompt:

>>> python run_nn_eeg.py


=========================

The project consists of several IPython notebooks, which carry out the exploratory data analysis and process results, and also three Python files which are called from the command line to actually run the model.

The project requires only standard Python libraries, including Numpy, Pandas, and Scikit-Learn

We now describe each of the files in order.


EEG_Eye_State.csv
=========================

This file was downloaded from the UCI Machine Learning repository. It contains approximately 15,000 samples of 14 numeric inputs and 1 binary label.


EEG_Eye_Exploration.ipynb
=========================

This notebook reads in the EEG_Eye_State.csv file and analyzes the statistics and distributions of the 14 numeric inputs and time series of the binary output. 


neuralnetwork.py
=========================


This file forms the core of this project. This file implements a neural network with a single hidden layer, and variable number of inputs and hidden units. 


cross_validation.py
=========================
The second file, cross_validation.py, is a helper file, with supporting functions.


run_nn_eeg.py
=========================
This file is called from the command line. Internally, it reads in the data file, creates a NeuralNetwork instance, and reports the classification metrics when the fitting is complete.


nn_eeg_results.ipynb
=========================
This notebook creates the graphs showing the learning curves reported by the neural network model.


EEG_randomforest.ipynb
=========================
This file independently analyzes the same dataset and reports classification accuracy for comparison with the neural network model created in this project.


Wmatrix_heatmap.ipynb
=========================
This notebook creates the heat map showing the coefficients of the weight matrix connecting the inputs to the hidden layer.
