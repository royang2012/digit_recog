# ======= enable this to run on GPU
import os
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

# =======
import pandas, numpy

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('th')
# =======
train_raw = numpy.array(pandas.read_csv('./input/train.csv'))

train_x = train_raw[:, 1:].reshape(train_raw.shape[0], 1, 28, 28).astype('float32') / 255.
train_y = np_utils.to_categorical(train_raw[:, 0], 10)

test_raw = numpy.array(pandas.read_csv('./input/test.csv'))
test_x = test_raw.reshape(test_raw.shape[0], 1, 28, 28).astype('float32') / 255.

# =======
model = Sequential()

model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(1, 28, 28)))
model.add(MaxPooling2D())

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D())

# model.add(Convolution2D(128, 3, 3, activation='relu'))


model.add(Flatten())
# model.summary()
model.add(Dense(output_dim=128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# =======
datagen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

# =======
num_diz = 20
prob_lst = []
for i in range(num_diz):  # change to 20
    prob = np.zeros([test_x.shape[0], 10])
    for j in range(12):
        model.fit(datagen.flow(train_x, train_y, batch_size=64), nb_epoch=10)  # change to 10
        prob += model.predict_proba(test_x)
    prob_lst.append(prob)
# model.save('model_mnist.h5') # uncomment to save your network

# y = model.predict_classes(test_x)
# numpy.savetxt('mnist.csv', np.c_[range(1, len(y) + 1), y], fmt='%d', delimiter=',', header='ImageId,Label')
predicted_class = np.zeros(test_x.shape[0])
prob_3d = np.dstack(prob_lst)
for i in range(0, prob_3d.shape[0]):
    conf_ratio = np.zeros(num_diz)
    for j in range(num_diz):
        two_choices = prob_3d[i, :, j].argsort()[-2:][::-1]
        conf_ratio[j] = prob_3d[i, two_choices[0], j]/prob_3d[i, two_choices[1], j]
    best_diz_idx = conf_ratio.argsort()[-10:][::-1]
    avg_best_diz = np.sum(prob_3d[i, :, best_diz_idx], axis=0)
    predicted_class[i] = np.argmax(avg_best_diz)
numpy.savetxt('mnist_161224.csv', np.c_[range(1, predicted_class.shape[0] + 1), predicted_class], fmt='%d', delimiter=',', header='ImageId,Label')
