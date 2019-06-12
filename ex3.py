import sys
import numpy as np
import random
from scipy.special import softmax
from numpy import inf


def create_x_data_set(train_path):
    '''
    get path to file and create data set from it.
    :param train_path: path to file.
    :return: data set.
    '''
    file = open(train_path)
    content = [line.rstrip('\n') for line in file]
    dataset = [line.split(" ") for line in content]
    dataset = [list(map(int, vector)) for vector in dataset]
    file.close()
    return dataset


def create_y_data_set(train_path):
    '''
    create y data set of floats from path.
    :param train_path: path
    :return: data set y of floats.
    '''
    file = open(train_path)
    content = [int(value) for value in file]
    file.close()
    return content


def shuffle_data(x, y):
    '''
    shuffle the data x and y accordingly.
    :param x: train set.
    :param y: cluster classification.
    :return: x and y shuffled.
    '''
    zip_x_y = list(zip(x, y))
    random.shuffle(zip_x_y)
    new_x, new_y = zip(*zip_x_y)
    return new_x, new_y


def init_weights(rows, cols=1):
    # if cols == 1:
    #     return np.random.rand(rows)
    # else:
    return np.random.randn(rows, cols)
    # return np.random.uniform(-0.08, 0.08, rows * cols).reshape(rows, cols)


def load_files():
    # train constants
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    train_X = np.loadtxt(train_x_path)
    train_Y = np.loadtxt(train_y_path)
    test_x = np.loadtxt(test_x_path)
    return train_X, train_Y, test_x


def train(train_x, train_y, lr, epochs, weights):
    '''
    sdfsdf.
    :param train_x:
    :param train_y:
    :param lr:
    :param epochs:
    :param weights:
    '''
    # final_acc = 0.0
    # iteration = 0
    for i in range(epochs):
        sum_loss = 0.0
        train_x, train_y = shuffle_data(train_x, train_y)
        for x, y in zip(train_x, train_y):
            fprop_cache = forward_prop(weights, x)
            loss = get_negative_log_loss(fprop_cache["y_hat"], y)
            sum_loss += loss
            gradients = back_prop(fprop_cache, y)
            weights = update_weights(weights, gradients, lr)
            # acc, validation_loss = validation_check(weights, validation_x, validation_y)
            # if final_acc < acc:
            #     final_acc = acc
            #     iteration = i
    # return final_acc, iteration
    # print(i, sum_loss / np.size(train_y), validation_loss, "{}%".format(acc * 100))
    return weights


def validation_check(weights, validation_x, validation_y):
    '''
    test the model each epoch on the validations set.
    :param weights: weights matrices.
    :param validation_x: validation data x.
    :param validation_y: classification data y.
    :return: correctness percent and average loss.
    '''
    sigma_loss = 0.0
    correct = 0.0
    num_of_examples = np.size(validation_x, 0)
    for x, y in zip(validation_x, validation_y):
        fprop_cache = forward_prop(weights, x)
        y_hat = fprop_cache["y_hat"]
        loss = get_negative_log_loss(y_hat, y)
        sigma_loss += loss
        if y_hat.argmax() == y:
            correct += 1
    accuracy = correct / num_of_examples
    avg_loss = sigma_loss / num_of_examples
    return accuracy, avg_loss


def predict(weights, test_x):
    '''
    predict the classifications of the data set x and output it to file.
    :param weights: weights matrices.
    :param test_x: validation data x.
    :return: nothing.
    '''
    file = open("test_y", "w")
    for x in test_x:
        fprop_cache = forward_prop(weights, x)
        y_hat = fprop_cache["y_hat"]
        classification = y_hat.argmax()
        file.write(str(classification) + "\n")
    file.close()


def relu(x):
    '''
    relu function get max between(0,x)
    :param x: sumple data.
    :return: max.
    '''
    return max(0, x)


def get_z2(weights, x):
    '''
    function compute part of the neural net.
    :param weights: list of weights: w1,w2 - matrices, b1,b2 - vectors.
    :param x: one data sampling.
    :return: w2*h+b2.
    '''
    w2 = weights[1]
    b2 = weights[3]
    h = get_h(weights, x)
    w2h = np.dot(w2, h)
    z2 = w2h + b2
    return z2


def get_h(weights, x):
    '''
    activate the activation function {ReLU,Sigmoid,TanH} to breake linearity.
    :return: result of g on z_1.
    :param weights: list of weights: w1,w2 - matrices, b1,b2 - vectors.
    :param x: one data sampling.
    '''
    g = np.vectorize(relu)
    z1 = get_z1(weights, x)
    h = g(z1)
    return h


def get_z1(weights, x):
    '''
    function compute part of the neural net.
    :param weights: list of weights: w1,w2 - matrices, b1,b2 - vectors.
    :param x: one data sampling.
    :return: w2*h+b2.
    '''
    w1 = weights[0]
    b1 = weights[2]
    z1 = np.dot(w1, x)
    z1 = np.add(z1, b1)
    return z1


def softMax(x):
    """Compute softmax values for each sets of scores in x.
    :param x: vector of features
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def forward_prop(weights, x):
    '''
    calculate the neural net functions.
    :param weights: list of weights: w1,w2 - matrices, b1,b2 - vectors.
    :param x: one data sampling.
    :return: y_hat = soft_max(z_2).
    '''
    w1, w2, b1, b2 = [weights[key] for key in ('w1', 'w2', 'b1', 'b2')]
    new_x = np.reshape(x, (-1, 1))
    z1 = np.dot(w1, new_x) + b1
    g = np.vectorize(relu)
    h = g(z1)
    maxi = h.max() or 1
    h = h / maxi
    z2 = np.dot(w2, h) + b2
    y_hat = softmax(z2)
    ret = {'x': new_x, 'z1': z1, 'h': h, 'z2': z2, 'y_hat': y_hat}
    for key in weights.keys():
        ret[key] = weights[key]
    return ret


def update_weights(weights, gradients, eta):
    '''
    the function update all the weights by the gradients and the eta
    :param gradients: gradients of the weights w1, w2, b1 b2.
    :param weights: matrix weights of the neural network.
    :param eta: learning rate of the neural network.
    :return: weights after updating.
    '''
    w1, w2, b1, b2 = [weights[key] for key in ('w1', 'w2', 'b1', 'b2')]
    dw1, dw2, db1, db2 = [eta * gradients[key] for key in ('dw1', 'dw2', 'db1', 'db2')]
    w1, w2, b1, b2 = w1 - dw1, w2 - dw2, b1 - db1, b2 - db2
    return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}


def relu_derivative(x):
    '''
    calculate relu derivative.
    :param x: paramter.
    :return: relu derivative 1 or 0.
    '''
    if x > 0:
        return 1
    return 0


def back_prop(fprop_cache, y):
    x, z1, h, z2, w2, y_hat = [fprop_cache[key] for key in ('x', 'z1', 'h', 'z2', 'w2', 'y_hat')]
    '''
    calculate w2:
    '''
    y_vec = np.zeros((y_hat.size, 1))
    y_vec[int(y)] = 1
    dl_dz2 = np.subtract(y_hat, y_vec)  # dL/dy_hat *_dy_hat/dz2
    dz2_dw2 = h.T  # dz2/dw2
    dw2 = np.dot(dl_dz2, dz2_dw2)  # dLw2= dL/dy_hat * dy_hat/dz2 * dz2/dw2

    '''
    calculate w1:
    '''
    dz2_dh = w2  # dz2/dh
    dl_dh = np.dot(dl_dz2.T, dz2_dh)  # dl/dh
    g = np.vectorize(relu_derivative)
    dh_dz1 = g(z1)  # dh1_dz1
    dz1_dw1 = x  # dz1/dw1
    # dw1 = np.dot(dl_dh.T, np.dot(dh_dz1.T, dz1_dw1))  # dLw1 = dL/dy_hat *_dy_hat/dz2 * dz2/dh * dh1/dz1 * dz1/dw1
    # dw1 = np.dot(dl_dh.T, np.dot(dh_dz1.T, dz1_dw1))  # dLw1 = dL/dy_hat *_dy_hat/dz2 * dz2/dh * dh1/dz1 * dz1/dw1
    dw1 = np.dot((dl_dh.T * dh_dz1), dz1_dw1.T)
    '''
    calculate b2:
    '''
    db2 = dl_dz2  # dz2/db2 = 1

    '''
    clculate b1:
    '''
    db1 = np.dot(dl_dh, dh_dz1)  # dz1/db1 = 1
    return {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}


def get_negative_log_loss(y_hat, y):
    '''
    calculate loss by negative log loss,with clusters vector: y_hat and the true classification y.
    :param y_hat: vector of percents clusters.
    :param y: real classification.
    :return: loss value.
    '''
    y_vec = np.zeros(y_hat.size)
    y_vec[int(y)] = 1
    # need to calculate sum of(y_i*log(y_hat_i))
    # first do log on y_hat:
    # TODO: check what to do with log on 0 (equal -inf). possible solution:
    # https://stackoverflow.com/questions/49602205/python-numpy-negative-log-likelihood-calculation-when-some-predicted-probabiliti
    y_hat = np.log2(y_hat)
    # change all -inf to zeros.
    y_hat[y_hat == -inf] = 0
    # return scalar of - y*log(y_hat)
    return -np.dot(y_vec, y_hat)


def normalize_data(x):
    '''
    the values is between 0-255, normalize the values to be [0,1]
    :param x: values to normalize.
    :return: x after normalize.
    '''
    x = np.divide(x, 255)
    return x


def check_hyper_parameters(lr, epochs, hidden_size, train_X, train_Y, clusters_num, validation_X, validation_Y):
    w1, w2 = init_weights(hidden_size, np.ma.size(train_X, 1)), init_weights(clusters_num, hidden_size)
    b1, b2 = init_weights(hidden_size), init_weights(clusters_num)
    weights = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return train(train_X, train_Y, lr, epochs, weights, validation_X, validation_Y)


def main():
    '''
    Hyper parameters:
    '''

    # number of learning iterations:
    epochs = 25
    # size of hidden layer:
    hidden_size = 90
    # number of clusters:
    clusters_num = 10
    # part size of validation from test:
    validation_percent = 0.2
    # learning rate: 0.1 or 0.01 or 0.001.
    lr = 0.1

    train_X, train_Y, test_X = load_files()
    train_X = normalize_data(train_X)
    # size of division to train and validation.
    # cross_validation_size = int(len(train_X) * validation_percent)

    # validation_X, validation_Y = train_X[cross_validation_size:], train_Y[cross_validation_size:]
    # train_X, train_Y = train_X[:cross_validation_size], train_Y[:cross_validation_size]

    '''
    init all the parameters, weight matrixes and vectors between layers with hidden layer size h.
    '''
    w1, w2 = init_weights(hidden_size, np.ma.size(train_X, 1)), init_weights(clusters_num, hidden_size)
    b1, b2 = init_weights(hidden_size), init_weights(clusters_num)
    weights = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    weights = train(train_X, train_Y, lr, epochs, weights)
    predict(weights, test_X)


if __name__ == "__main__":
    main()
