#Mnist-DNN
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
import numpy as np
#数据预处理
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],-1)/255
x_test=x_test.reshape(x_test.shape[0],-1)/255
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#神经网络的建设
model=Sequential([Dense(32,input_dim=784),Activation("relu"),Dense(10),Activation("softmax")])
rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-8,decay=0.0)
model.compile(optimizer=rmsprop,loss="categorical_crossentropy",metrics=["accuracy"])
#开始训练
model.fit(x_train,y_train,nb_epoch=2,batch_size=32)
loss,accuracy=model.evaluate(x_test,y_test)
print("loss",loss)
print("test accuracy",accuracy)


#Mnist-CNN
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
#数据的预处理
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,1,28,28)
x_test=x_test.reshape(-1,1,28,28)
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#构建网络模型
model=Sequential()
model.add(Convolution2D(32,5,5,border_mode="same",input_shape=(1,28,28)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="same"))

model.add(Convolution2D(64,5,5,border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),border_mode="same"))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
adam=Adam(lr=1e-4)

model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=["accuracy"])

#训练数据
model.fit(x_train,y_train,nb_epoch=1,batch_size=32)

loss,accuracy=model.evaluate(x_test,y_test)
print("loss",loss)
print("accuracy",accuracy)


#Mnist-RNN
