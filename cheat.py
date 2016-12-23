import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
train_data_raw = pd.read_csv("./input/train.csv").values
test_data_raw = pd.read_csv("./input/test.csv").values

img_cols = 28
img_rows = 28

train_X = train_data_raw[:, 1:].reshape(train_data_raw.shape[0], 1, img_rows, img_cols)



plt.subplot(10, 1, 1)
plt.imshow(train_X[0][0])
plt.subplot(10, 1, 2)
plt.imshow(train_X[1][0])
plt.subplot(10, 1, 3)
plt.imshow(train_X[2][0])
plt.subplot(10, 1, 4)
plt.imshow(train_X[3][0])
plt.subplot(10, 1, 5)
plt.imshow(train_X[4][0])
plt.subplot(10, 1, 6)
plt.imshow(train_X[5][0])
plt.subplot(10, 1, 7)
plt.imshow(train_X[6][0])
plt.subplot(10, 1, 8)
plt.imshow(train_X[7][0])
plt.subplot(10, 1, 9)
plt.imshow(train_X[8][0]) 
plt.subplot(10, 1, 10)
plt.imshow(train_X[9][0])

train_X_0 = train_X[0][0].reshape(28*28, 1)
diff = train_data_raw[0, 1:] - train_X_0
diffn = np.array(diff)
print(np.count_nonzero(diffn))




plt.show()
