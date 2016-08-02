import numpy as np

class NeuralNetwork():

    def __init__(self,
                 num_hidden_units = 4,
                 learning_rate = 0.1,
                 L2_reg = 0,
                 n_iter = 100,
                 n_print = 100,
                 seed = 42,
                 precision = 4,
                 verbose = True):
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.L2_reg = L2_reg
        self.n_iter = n_iter
        self.n_print = n_print
        np.random.seed(seed)
        np.set_printoptions(precision=precision, suppress=True)
        self.verbose = verbose

        self.W = None
        self.V = None
        self.num_examples = None
        self.num_inputs = None
        self.num_outputs = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

  
    # initializations of neural net weights
    def _initialize_weights(self):
        self.W = np.random.randn(self.num_hidden_units,\
                 self.num_inputs + 1) # 1st layer weights, add one for bias unit
        self.V = np.random.randn(self.num_outputs,\
                 self.num_hidden_units + 1) # output layer weights, add one for bias unit

    def _shuffle_examples(self, X, y):
        ind = np.random.permutation(len(y))
        return X[ind], y[ind]

    def _add_bias_units_to_matrix(self, mat):
        num_rows = mat.shape[0]
        mat = np.concatenate((mat, np.ones((num_rows,1)) ), axis=1)
        return mat

    def _add_bias_unit_to_array(self, arr):
        bias_unit = np.array([1])
        arr = np.concatenate( (arr, bias_unit), axis = 0)
        return arr

    def fit(self, X, y):
        self.num_examples = X.shape[0]
        self.num_inputs = X.shape[1]
        self.num_outputs = y.shape[1]
        self.num_labelled_outputs = y.shape[0]
        self._initialize_weights()
        self.mse = []
        print "Number of Examples found: {}".format(self.num_examples)
        print "Number of Inputs found: {}".format(self.num_inputs)
        print "Number of outputs found: {}".format(self.num_outputs)
        if self.num_examples != self.num_labelled_outputs:
            raise Exception("Number of inputs doesn't match number of outputs")
        iter_print = max(self.n_iter / 100, 1)

        X = self._add_bias_units_to_matrix(X)

        for iter in xrange(self.n_iter):
            if iter % self.n_print == 0:         
                print "iter = {}".format(iter)

            error = 0
            # randomly reorder all examples each iteration
            X,y = self._shuffle_examples(X,y)
            for i in xrange(len(X)):

                # implement feed forward network
                z = np.dot(self.W,X[i].T) # activation vector input to hidden layer
                u = self.sigmoid(z)       # activation vector of hidden layer

                u = self._add_bias_unit_to_array(u)  # add internal bias unit to hidden units

                f = np.dot(self.V,u)    # activation scalar input to output layer
                h = self.sigmoid(f)      # activation scalar of output layer

                # begin backpropagation
                regW = -self.L2_reg * self.W       # L2 regularization
                regV = -self.L2_reg * self.V       # L2 regularization
                
                # chain rule gives derivative of error with respect to V matrix
                dEdV =  regV + np.outer( (y[i]-h) * self.sigmoid_derivative(h), u)
                dEdW = regW

                # sum errors over each output if more than one
                for k in xrange(self.num_outputs):
                    # chain rule gives derivative of error with respect to W matrix
                    dEdW = (y[i][k]-h[k]) * self.sigmoid_derivative(h[k]) *\
                             np.outer(self.V[k][:-1] *  self.sigmoid_derivative(u[:-1]), X[i])
                
                self.V += self.learning_rate * dEdV
                self.W += self.learning_rate * dEdW

                error += np.sum( (h - y[i]) * (h - y[i]) )
                if self.verbose == True and iter % self.n_print == 0:         
                    print "iter = {}, input = {}, predicted_output = {}, actual_output = {}, error = {}"\
                                .format(iter, X[i], h, y[i], h - y[i])

            if iter % self.n_print == 0:
                self.mse.append(error)
        
        
        print "self.mse = {}".format(self.mse)

    # calculate binary output labels based on nn weights and input X
    def predict(self, X, threshold = 0.5):
        X = self._add_bias_units_to_matrix(X)
        prob = self.predict_prob(X)
        prob[prob >= threshold] = 1
        prob[prob < threshold] = 0
        return prob

    # calculate probabilities output label = 1 based on nn weights and input X
    def predict_prob(self, X, add_bias_unit=False):
        if add_bias_unit == True:
            X = self._add_bias_units_to_matrix(X)
        z = np.dot(X, self.W.T) # activation vector input to hidden layer
        u = self.sigmoid(z)       # activation vector of hidden layer
        u = self._add_bias_units_to_matrix(u)  # add internal bias unit to hidden units
        f = np.dot(u,self.V.T)    # activation scalar input to output layer
        h = self.sigmoid(f)      # activation scalar of output layer

        return h

