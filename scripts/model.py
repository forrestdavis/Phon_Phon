#########################################################
# This is a modified version of the example from
# deeplearning.net that classifies MNSIT digits using
# Logistic Regression
#########################################################
from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import matplotlib.pyplot as plt

import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # Initialize a matrix of dimensions n_in * n_out with all zeros, 
        # set to be weights (W)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ), name='W', borrow=True)

        # Initialize a vector of length n_out with all zeros, 
        # set to be biases (linear affix) (b) 
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ), name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))

        else:
            raise NotImplementedError()

#Load data representing only the last sound in the stem
def load_data_last_sound(data_dir):

    rval = []

    '''
    train_data = open(data_dir+'train.data', 'r')
    dev_data = open(data_dir+'dev.data', 'r')
    test_data = open(data_dir+'test.data', 'r')
    '''

    #train_data = open(data_dir+'class_1/train_f.data', 'r')
    train_data = open(data_dir+'train.data', 'r')
    dev_data = open(data_dir+'dev.data', 'r')
    test_data = open(data_dir+'test.data', 'r')
    #test_data = open(data_dir+'voiceless_test.data', 'r')

    #Get x and y train values
    temp_x = []
    temp_y = []
    for line in train_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for last sound
        x_val = line[len(line)-2:len(line)-1]
        x_val = x_val[0].split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    train_data_x = numpy.array(temp_x)
    train_data_y = numpy.array(temp_y)
    train_set = (train_data_x, train_data_y)
         
    #Get x and y dev values
    temp_x = []
    temp_y = []
    for line in dev_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for last sound
        x_val = line[len(line)-2:len(line)-1]
        x_val = x_val[0].split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    dev_data_x = numpy.array(temp_x)
    dev_data_y = numpy.array(temp_y)
    dev_set = (dev_data_x, dev_data_y)

    #Get x and y test values
    temp_x = []
    temp_y = []
    for line in test_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for last sound
        x_val = line[len(line)-2:len(line)-1]
        x_val = x_val[0].split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    test_data_x = numpy.array(temp_x)
    test_data_y = numpy.array(temp_y)
    test_set = (test_data_x, test_data_y)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    dev_set_x, dev_set_y = shared_dataset(dev_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (dev_set_x, dev_set_y),
            (test_set_x, test_set_y)]

    #Clean it up
    train_data.close()
    dev_data.close()
    test_data.close()
    return rval

#Load data representing the full stem (padded to max length)
def load_data_full(data_dir):

    train_data = open(data_dir+'train_pad.data', 'r')
    dev_data = open(data_dir+'dev_pad.data', 'r')
    test_data = open(data_dir+'test_pad.data', 'r')

    #Get x and y train values
    temp_x = []
    temp_y = []
    for line in train_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for all sounds
        x_val = line[2:len(line)-1]

        #Join all x values
        x_val = ' '.join(x_val)
        x_val = x_val.split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]
        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    train_data_x = numpy.array(temp_x)
    train_data_y = numpy.array(temp_y)
    train_set = (train_data_x, train_data_y)

    #Get x and y dev values
    temp_x = []
    temp_y = []
    for line in dev_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for all sounds
        x_val = line[2:len(line)-1]

        #Join all x values
        x_val = ' '.join(x_val)
        x_val = x_val.split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    dev_data_x = numpy.array(temp_x)
    dev_data_y = numpy.array(temp_y)
    dev_set = (dev_data_x, dev_data_y)

    #Get x and y test values
    temp_x = []
    temp_y = []
    for line in test_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for all sounds
        x_val = line[2:len(line)-1]

        #Join all x values
        x_val = ' '.join(x_val)
        x_val = x_val.split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))

    test_data_x = numpy.array(temp_x)
    test_data_y = numpy.array(temp_y)
    test_set = (test_data_x, test_data_y)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    dev_set_x, dev_set_y = shared_dataset(dev_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (dev_set_x, dev_set_y),
            (test_set_x, test_set_y)]

    #Clean it up
    train_data.close()
    dev_data.close()
    test_data.close()
    return rval


#Gradient descent 
def sgd_optimization(data_type, learning_rate=0.13, 
                           n_epochs=1000,
                           data_dir = '../data_ternary/',
                           batch_size=300):

    if data_type == 'full':
        datasets = load_data_full(data_dir)
    else:
        datasets = load_data_last_sound(data_dir)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    if data_type == 'full':
        classifier = LogisticRegression(input=x, n_in=20*17, n_out=3)
    else:
        classifier = LogisticRegression(input=x, n_in=20*1, n_out=3)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 3000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.05  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    all_costs = []
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        mini_costs = []
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            #Maybe
            mini_costs.append(minibatch_avg_cost.tolist())

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('../saved_models/full_ternary.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

        all_costs.append(sum(mini_costs)/len(mini_costs))
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            ' with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
    return classifier.W.get_value(), all_costs

def predict():

    # load the saved model
    classifier = pickle.load(open('../saved_models/full_ternary.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    data_dir = "../data_ternary/"
    datasets = load_data_last_sound(data_dir)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    s = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0,
            0, 0, -1, 0, 1, 0, 0, 1, -1, 1]
    z = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0,
            0, 0, 1, 0, 1, 0, 0, 1, -1, 1]
    S = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 
            0, 0, -1, 0, 1, 0, 0, -1, 1, 1]
    Z = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 
            0, 0, 1, 0, 1, 0, 0, -1, 1, 1]
    tS = [0, 0, 0, 0, 0, -1, 1, -1, -1, 0, 
            0, 1, -1, 0, 1, 0, 0, -1, 1, 1]
    dZ = [0, 0, 0, 0, 0, -1, 1, -1, -1, 0, 
            0, 1, 1, 0, 1, 0, 0, -1, 1, 1]

    p = [0, 0, 0, 0, 0, -1, 1, -1, -1, 0, 
            0, 0, -1, 1, 0, 0, 0, 0, 0, 0]
    t = [0, 0, 0, 0, 0, -1, 1, -1, -1, 0, 
            0, 0, -1, 0, 1, 0, 0, 1, -1, 0]
    k = [1, 1, -1, 0, 0, -1, 1, -1, -1, 0, 
            0, 0, -1, 0, 0, 1, 0, 0, 0, 0]
    f = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 
            0, 0, -1, 1, 0, 0, 0, 0, 0, 1]
    T = [0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 
            0, 0, -1, 0, 1, 0, 0, 1, 1, -1]

    predicted_values = predict_model(test_set_x)

    count_p = [0, 0, 0, 0]
    count_t = [0, 0, 0, 0]
    count_k = [0, 0, 0, 0]
    count_f = [0, 0, 0, 0]
    count_T = [0, 0, 0, 0]

    for x, y in zip(test_set_x, predicted_values):
        if numpy.array_equal(x, p):
            if y == 0:
                count_p[0] += 1
            if y == 1:
                count_p[1] += 1
            if y == 2:
                count_p[2] += 1
            count_p[3] += 1

        if numpy.array_equal(x, t):
            if y == 0:
                count_t[0] += 1
            if y == 1:
                count_t[1] += 1
            if y == 2:
                count_t[2] += 1
            count_t[3] += 1

        if numpy.array_equal(x, k):
            if y == 0:
                count_k[0] += 1
            if y == 1:
                count_k[1] += 1
            if y == 2:
                count_k[2] += 1
            count_k[3] += 1

        if numpy.array_equal(x, f):
            if y == 0:
                count_f[0] += 1
            if y == 1:
                count_f[1] += 1
            if y == 2:
                count_f[2] += 1
            count_f[3] += 1

        if numpy.array_equal(x, T):
            if y == 0:
                count_T[0] += 1
            if y == 1:
                count_T[1] += 1
            if y == 2:
                count_T[2] += 1
            count_T[3] += 1

    print('sound', '0', '1', '2', 'total')
    print('p', count_p[0], count_p[1], count_p[2], count_p[3])
    print('t', count_t[0], count_t[1], count_t[2], count_t[3])
    print('k', count_k[0], count_k[1], count_k[2], count_k[3])
    print('f', count_f[0], count_f[1], count_f[2], count_f[3])
    print('T', count_T[0], count_T[1], count_T[2], count_T[3])

def stats():

    #load saved model
    classifier = pickle.load(open('../saved_models/full_ternary.pkl'))
    weights = classifier.W.get_value()

    #Get each weight
    w_1 = []
    w_2 = []
    w_3 = []
    for w in weights:
        w_1.append(w[0])
        w_2.append(w[1])
        w_3.append(w[2])
    a_1 = numpy.array(w_1)
    a_2 = numpy.array(w_2)
    a_3 = numpy.array(w_3)

    #Detect outliers
    def detect_outliers(values):

        threshold = 3.0

        median = numpy.median(values)
        MAD = numpy.median(
                [numpy.abs(value-median) for value in values])
        modified_z_scores = [1.4826 * (value - median) 
                / MAD for value in values]
        return [numpy.where(numpy.abs(modified_z_scores) > threshold),
                modified_z_scores]

    a_1_outliers, a_1_z_scores = detect_outliers(a_1)
    a_2_outliers, a_2_z_scores = detect_outliers(a_2)
    a_3_outliers, a_3_z_scores = detect_outliers(a_3)

    features = ['High', 'Back', 'Low', 'ATR', 'Round', 
            'Syllabic', 'Cons',
            'Son', 'Cont', 'Nasal', 'Lateral', 'DR', 'Voice', 
            'Labial', 'Coronal', 'Dorsal', 'Laryngeal', 'Anterior',
            'Dist', 'Strident']

    a_1_outliers = a_1_outliers[0].tolist()

    pairs = []
    for outlier in a_1_outliers:
        pairs.append((outlier, a_1_z_scores[outlier]))
    pairs.sort(key=lambda tup: abs(tup[1]), reverse=True)


    pairs = pairs[:5]
    print(pairs)
    x_values = []
    y_values = []
    for pair in pairs:
        x_values.append(features[pair[0]%20])
        y_values.append(abs(pair[1]))

    y_pos = numpy.arange(len(x_values))

    plt.bar(y_pos, y_values, align='center', alpha=0.5, color='r')
    plt.xticks(y_pos, x_values)
    plt.xlabel('Feature')
    plt.ylabel('Standard Deviations from the Median')
    plt.title('Relative Feature Prominence for type 0')
    plt.show()

    a_2_outliers = a_2_outliers[0].tolist()

    pairs = []
    for outlier in a_2_outliers:
        pairs.append((outlier, a_2_z_scores[outlier]))
    pairs.sort(key=lambda tup: abs(tup[1]), reverse=True)


    pairs = pairs[:5]
    print(pairs)
    x_values = []
    y_values = []
    for pair in pairs:
        x_values.append(features[pair[0]%20])
        y_values.append(abs(pair[1]))

    y_pos = numpy.arange(len(x_values))

    plt.bar(y_pos, y_values, align='center', alpha=0.5, color='r')
    plt.xticks(y_pos, x_values)
    plt.xlabel('Feature')
    plt.ylabel('Standard Deviations from the Median')
    plt.title('Relative Feature Prominence for type 1')
    plt.show()

    a_3_outliers = a_3_outliers[0].tolist()

    pairs = []
    for outlier in a_3_outliers:
        pairs.append((outlier, a_3_z_scores[outlier]))
    pairs.sort(key=lambda tup: abs(tup[1]), reverse=True)


    pairs = pairs[:5]
    print(pairs)
    x_values = []
    y_values = []
    for pair in pairs:
        x_values.append(features[pair[0]%20])
        y_values.append(abs(pair[1]))

    y_pos = numpy.arange(len(x_values))

    plt.bar(y_pos, y_values, align='center', alpha=0.5, color='r')
    plt.xticks(y_pos, x_values)
    plt.xlabel('Feature')
    plt.ylabel('Standard Deviations from the Median')
    plt.title('Relative Feature Prominence for type 2')
    plt.show()

if __name__ == '__main__':

    #weights, costs = sgd_optimization('full')
    weights, costs = sgd_optimization('final')

    stats()
    #predict()

    if 1:
        plt.plot(costs)
        plt.xlabel('Training Iterations')
        plt.ylabel('Cost')
        plt.title('Training Costs')
        plt.show()
