
import numpy as np
from keras.src.losses import SparseCategoricalCrossentropy
from tensorflow.keras import models

# (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number).
Y_FEATURE = 4

# data = np.load("processed_data.npy")
data = np.load("processed_data.npy")
print(np.shape(data))

# for sample in data[:, :, 0:2]:
#     x = sample[:,0]
#     y = sample[:,1]
#     plt.plot(x, y, marker="o")
#     plt.show()

split_index = int(0.8*len(data))
train_data = data[:split_index]
val_data = data[split_index:]

x_train = train_data[:, :, 0:2]
y_train = train_data[:, :, Y_FEATURE]
x_val = val_data[:, :, 0:2]
y_val = val_data[:, :, Y_FEATURE]

y_train = np.squeeze(y_train[:, 0])
y_val = np.squeeze(y_val[:, 0])

model = models.load_model("loops_run_1/file_processed_data_linear_standard_num_0_feature_2_1D_20Dense_2LSTM_64LSTM_50epochs_val_acc_67.49_17_47_32.keras")
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

print('herew')
sample1 = 34
sample2 = 66
x_sample_1 = x_train[sample1]
x_sample_2 =  x_train[sample2]
print(np.shape(x_sample_1))
prediction = model.predict(np.array([x_sample_1, x_sample_2]))

print(y_train[sample1])
print(y_train[sample2])
print(np.argmax(prediction[0]))
print(np.argmax(prediction[1]))

