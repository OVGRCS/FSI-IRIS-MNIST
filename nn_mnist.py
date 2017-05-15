import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

#Obtencion de datos mediante cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Set de datos y parseo

train_x, train_y= train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int),10)
valid_y = one_hot(valid_y.astype(int),10)
test_y = one_hot(test_y.astype(int),10)

#Declaracion de los placeholders y las variables

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(np.float32(np.random.rand(784,10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10,10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#Optimizadores

loss = tf.reduce_sum(tf.square(y_ - y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train = tf.train.RMSPropOptimizer(0.001).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Inicio de la sesion de TensorFlow

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#Comienzo del entrenamiento de la red

print "---------------------"
print "   Start Training... "
print "---------------------"

batch_size = 100
plotError = []
errorA = 15000
errorB = 15000
epoch = 0

while errorA <= errorB:
    for jj in xrange(len(train_x) / batch_size):
       batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
       batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
       sess.run(train, feed_dict={x: batch_xs, y_:batch_ys})
    epoch += 1
    #Cambiamos la comparativa del error segun el optimizador elegido
    errorB = errorA
    errorA = sess.run(loss, feed_dict={x: valid_x, y_:valid_y})
    #errorA = sess.run(cross_entropy, feed_dict={x: valid_x, y_: valid_y})
    plotError.append(errorA)
    print "Epoch #: ", epoch, " Error: ", errorA
    print "-------------------------------------------------------------"

#Prueba con datos nuevos

print "---------------------"
print "    Testing Area     "
print "---------------------"

result = sess.run(y, feed_dict={x: test_x})
errorcount = 0
for b, r in zip (test_y, result):
    if np.argmax (b) != np.argmax (r):
        errorcount += 1
print "No. of errors: ", errorcount
print "-------------------------------------------------------------"

plt.ylabel('Errores')
plt.xlabel('Iteraciones')
plt.title('RMS Prop Optimizer 0.001 Training')
xLine = plt.plot(plotError)
plt.legend(handle=[xLine])
plt.show()

