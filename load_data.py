from __future__ import print_function
import theano
import theano.tensor as T
import numpy 

def load_data(data_dir):

    rval = []

    train_data = open(data_dir+'train.data', 'r')
    dev_data = open(data_dir+'dev.data', 'r')
    test_data = open(data_dir+'test.data', 'r')

    #Get x and y values
    temp_x = []
    temp_y = []
    for line in train_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get x values (binary feature values)
        x_val = line[2:len(line)-1]

        #Take list and break strings into lists
        break_list = lambda x: x.split()
        x_val_split = list(map(break_list, x_val))

        #Turn these list of lists containing strings to ints
        x_val = [[int(y) for y in x] for x in x_val_split]

        temp_x.append(x_val)
        temp_y.append(int(line[len(line)-1]))
    print(temp_x)
    print(temp_y)
         

    '''
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
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    '''

    #Clean it up
    train_data.close()
    dev_data.close()
    test_data.close()
    return rval

if __name__ == "__main__":

    data_dir = "data/"
    rval = load_data(data_dir)
