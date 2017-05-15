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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:104, 0:4].astype('f4')  # the samples are the four first rows of data
x_data15 = data[105:127, 0:4].astype('f4')
x_dataTest = data[128:, 0:4].astype('f4')

y_data = one_hot(data[:104, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
y_data15 = one_hot(data[105:127, 4].astype(int), 3)
y_dataTest = one_hot(data[128:, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 22
plotError = []

for epoch in xrange(100):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    error = sess.run(loss, feed_dict={x: x_data15, y_: y_data15})
    plotError.append(error)
    print "Epoch #:", epoch, "Error: ", error
    result = sess.run(y, feed_dict={x: x_data15})
    for b, r in zip(y_data15, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"

#for jj in xrange(len(x_data15) / batch_size):

#    pruebax = x_data15[jj * batch_size: jj * batch_size + batch_size]
#    pruebay = y_data15[jj * batch_size: jj * batch_size + batch_size]

print "Error: ", sess.run(loss, feed_dict={x: x_dataTest, y_: y_dataTest})
aux = sess.run(y, feed_dict={x: x_dataTest})

for b, r in zip(y_dataTest, aux):
    print b, "-->", r
print "-----------------------------------------------------------------------------"

plt.ylabel('Errores')
plt.xlabel('Iteraciones')
plt.title('Gradient Optimizer 0.01 LR Training')
xLine = plt.plot(plotError)
plt.legend(handle=[xLine])
plt.show()