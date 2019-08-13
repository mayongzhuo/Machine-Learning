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

        
#Mnist-CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist=input_data.read_data_sets("D:/MNIST",one_hot=True)

#数据预处理
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#参数赋值
batch_size=100
n_batch=mnist.train.num_examples//batch_size

#创建网络
w_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
conv2d_1=tf.nn.conv2d(x_image,w_conv1,strides=[1,1,1,1],padding="SAME")
output_1=tf.nn.max_pool(tf.nn.relu(conv2d_1),ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
conv2d_2=tf.nn.conv2d(output_1,w_conv2,strides=[1,1,1,1],padding="SAME")
output_2=tf.nn.max_pool(tf.nn.relu(conv2d_2),ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

h_pool2_flat=tf.reshape(output_2,[-1,7*7*64])
wfc_1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
bfc_1=tf.Variable(tf.constant(0.1,shape=[1024]))
wx_plus_b1=tf.matmul(h_pool2_flat,wfc_1)+bfc_1
h_fc1=tf.nn.relu(wx_plus_b1)
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

wfc_2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
bfc_2=tf.Variable(tf.constant(0.1,shape=[10]))
wx_plus_b2=tf.matmul(h_fc1_drop,wfc_2)+bfc_2
prediction=tf.nn.softmax(wx_plus_b2)

#损失函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#优化器优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#求准确率
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1001):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        if i%100==0:
            tset_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            print(tset_acc)
