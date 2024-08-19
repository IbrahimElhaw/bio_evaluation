import random
from datetime import datetime

import numpy as np

from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, LambdaCallback
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.losses import SparseCategoricalCrossentropy

# (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number).
Y_FEATURE = 3

random.seed(10)

data = np.load("processed_data.npy")
print(np.shape(data))
random.shuffle(data)
# data = data[:len(data) // 2]
random.shuffle(data)

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

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_val))
print(np.shape(y_val))

counts = np.bincount(np.array(y_train, dtype=int))
count_zeros = counts[0]
count_ones = counts[1]
print(f"Number of 0s: {count_zeros}")
print(f"Number of 1s: {count_ones}")

num_classes = len(np.unique(y_train))
print("num of classes ", num_classes)

model = Sequential()
model.add(Input(shape=np.shape(x_train[0])))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

desired_accuracy = 0.99

accuracy_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: (
        logs.get('val_accuracy') is not None and logs['val_accuracy'] >= desired_accuracy
    ) and (
        print(f"Reached {desired_accuracy*100:.2f}% accuracy, stopping training."),
        setattr(model, 'stop_training', True)  # Explicitly stop training
    )
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, accuracy_callback]
)

final_val_accuracy = history.history['val_accuracy'][-1] * 100
current_time = datetime.now().strftime("%H_%M_%S")
file_name = f"feature_{Y_FEATURE}, model_{current_time}_val_acc_{final_val_accuracy:.2f}.keras"
model.save(file_name)
