import pandas as pd
import numpy as np
import keras.layers.core as kcore
import keras.layers.convolutional as kconv
import keras.layers.pooling as kpool
import keras.models as kmodels
import keras.utils.np_utils as kutils

# read data from hard drive
train_data_raw = pd.read_csv("./input/train.csv").values
test_data_raw = pd.read_csv("./input/test.csv").values

img_cols = 28
img_rows = 28

train_X = train_data_raw[:, 1:].reshape(train_data_raw.shape[0], 1, img_rows, img_cols)
train_Y = kutils.to_categorical(train_data_raw[:, 0])
num_class = train_Y.shape[1]

num_filters_1 = 32
conv_dim = 3
cnn = kmodels.Sequential()
cnn.add(kconv.ZeroPadding2D((1,1), input_shape=(1, 28, 28),))
cnn.add(kconv.Convolution2D(num_filters_1, conv_dim, conv_dim,  activation="relu"))
cnn.add(kpool.MaxPooling2D(strides=(2, 2)))

num_filters_2 = 64
cnn.add(kconv.ZeroPadding2D((1, 1)))
cnn.add(kconv.Convolution2D(num_filters_2, conv_dim, conv_dim, activation="relu"))
cnn.add(kpool.MaxPooling2D(strides=(2, 2)))

cnn.add(kconv.ZeroPadding2D((1, 1)))
cnn.add(kcore.Flatten())
cnn.add(kcore.Dropout(0.2))
cnn.add(kcore.Dense(128, activation="relu")) # 4096
cnn.add(kcore.Dense(num_class, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
cnn.fit(train_X, train_Y, batch_size=256, nb_epoch=50, verbose=1)

test_X = test_data_raw.reshape(test_data_raw.shape[0], 1, 28, 28)
test_X = test_X.astype(float)
test_X /= 255.0

yPred = cnn.predict_classes(test_X)

np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
kmodels.save("first_model.h5")
