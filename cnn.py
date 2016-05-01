import numpy as np
import os
import theano
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import glob


sng = RandomStreams(23455)

x = T.tensor4()
t = T.matrix()

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data_dir = "cifar-10-batches-py"
class_names_cifar10 = np.load(os.path.join(data_dir, "batches.meta"))


def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def load_batch_cifar10(filename, dtype):
    path = os.path.join(data_dir, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 - 0.5
    labels = one_hot(batch['labels'], n=10)
    return data.astype(dtype), labels.astype(dtype)

def gray(x):
    return x.reshape(x.shape[0], 3, 32, 32).mean(1).reshape(x.shape[0], -1)


def cifar10(dtype='float64', grayscale=True):
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = load_batch_cifar10("data_batch_%d" %(k+1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)
    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)
    x_test, t_test = load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = gray(x_train)
        x_test = gray(x_test)

    return x_train, x_test, t_train, t_test

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(Shape):
    return theano.shared(floatX(np.random.rand(*Shape)*0.01))

def up_Pram(cost, param, lr, rho):
    grads = T.grad(cost, param)
    updates = []
    for a, b in zip(param, grads):
        acc = theano.shared(np.zeros(a.get_value().shape, dtype=theano.config.floatX))
        p = rho*acc - lr*b
        updates.append((acc, p))
        updates.append((a, a+p))
    return updates

def model(x, w1, b1, w2, b2, w3, b3, w, b):
    cov1 = T.maximum(0, conv2d(x, w1)+b1.dimshuffle('x', 0, 'x', 'x'))
    pool1 = max_pool_2d(cov1, (2, 2))
    cov2 = T.maximum(0, conv2d(pool1, w2)+b2.dimshuffle('x', 0, 'x', 'x'))
    pool2 = max_pool_2d(cov2, (2, 2))
    pool2_flat = pool2.flatten(2)
    h3 = T.maximum(0, T.dot(pool2_flat, w3) + b3)
    predict_y = T.nnet.softmax(T.dot(h3, w) + b)
    return predict_y

w1 = init_weights((4, 3, 3, 3))
b1 = init_weights((4,))
w2 = init_weights((8, 4, 3, 3))
b2 = init_weights((8,))
w3 = init_weights((8*4*4, 100))
b3 = init_weights((100,))
w = init_weights((100, 10))
b = init_weights((10,))

params = [w1, b1, w2, b2, w3, b3, w, b]

x_train, x_test, t_train, t_test = cifar10(dtype=theano.config.floatX, grayscale=False)
labels_test = np.argmax(t_test, axis=1)

predict_y = model(x, *params)
y = T.argmax(predict_y, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(predict_y, t))

updates = up_Pram(cost, params, 0.01, 0.9)

train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)

batch_size = 50

for i in range(5):
    print "iteration %d" %(i+1)
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = train(x_batch, t_batch)

    predict_test = predict(x_test)
    accuracy = np.mean(predict_test == labels_test)
    print "accuracy: %.5f" % accuracy
    print