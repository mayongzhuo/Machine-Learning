#Mnist-DNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist=input_data.read_data_sets("D:/MNIST",one_hot=True)

#变量的赋值
batch_size=100
n_batch=mnist.train.num_examples//batch_size

#网络搭建
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
weight=tf.Variable(tf.zeros([784,10]))
bias=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,weight)+bias)

#定义损失函数
loss=tf.reduce_mean(tf.square(y-prediction))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()

#给网络填入数据进行训练
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        corret_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
        accuracy=tf.reduce_mean(tf.cast(corret_prediction,tf.float32))
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("第%s轮，精度为%s"%(epoch,acc))
