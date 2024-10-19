import os
import random
from datetime import datetime
import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.losses import SparseCategoricalCrossentropy

# (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number, 8:time).


def calculate_velocity(x_data, time_data, threshold=3):
    # Calculate differences
    delta_x = np.diff(x_data[:, :, 0], axis=1)
    delta_y = np.diff(x_data[:, :, 1], axis=1)
    distances = np.sqrt(delta_x ** 2 + delta_y ** 2)
    delta_time = np.diff(time_data, axis=1)

    # Replace non-positive delta_time values with the previous value
    mask = delta_time <= 0
    for i in range(delta_time.shape[0]):
        for j in range(1, delta_time.shape[1]):
            if mask[i, j]:
                delta_time[i, j] = delta_time[i, j - 1]

    # Calculate velocities
    velocities = distances / delta_time

    # Add zero velocity at the last time step to match the shape
    velocities = np.concatenate([velocities, np.zeros((velocities.shape[0], 1))], axis=1)

    # Expand dimensions to match the shape of x_data
    velocities = np.expand_dims(velocities, axis=-1)

    # Flatten velocities for outlier detection
    velocities_flattened = velocities.reshape(-1, 1)

    # Calculate median and MAD (Median Absolute Deviation)
    median = np.median(velocities_flattened)
    mad = np.median(np.abs(velocities_flattened - median))

    # Define outlier thresholds
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad

    # Clip values to be within bounds
    velocities_clipped = np.clip(velocities_flattened, lower_bound, upper_bound)
    velocities_filtered = velocities_clipped.reshape(velocities.shape)

    return velocities_filtered


def save_model(model_, history_, shape_=None ):
    final_val_accuracy = history_.history['val_accuracy'][-1] * 100
    current_time = datetime.now().strftime("%H_%M_%S")
    folder_path = "loops_run_1"
    suffix = ""
    if shape_ is not None and shape_[2] == 4:
        suffix = "_with_V"
    file_name = (
        f"{folder_path}/{final_val_accuracy:.2f}_"
        f"file_{os.path.splitext(file)[0]}_num_{CURRENT_NUMBER}_feature_{Y_FEATURE}_"
        f"{NumberOfDense}D_{NeuronOfDenseCells}Dense_"
        f"{NumberOfLSTM}LSTM_{NeuronOfLSTMCells}LSTM_{EPOCHES}epochs_"
        f"val_acc_{final_val_accuracy:.2f}_{current_time}{suffix}.keras"
    )
    model_.save(file_name)


if __name__ == '__main__':
    # define some Constants for training model and wanted features
    # (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number, 8:time).
    Y_FEATURE = 2
    CURRENT_NUMBER = 6
    CURRENT_HAND = 1   # 1: right, 0: left
    CURRENT_FINGER = 1   # 1: index, 0: thumb
    NumberOfDense = 1
    NumberOfLSTM = 2
    NeuronOfDenseCells = 16
    NeuronOfLSTMCells = 64
    EPOCHES = 30
    random.seed(10)
    file = "processed_data_cubic_standard_with_time.npy"

    # load data and filter the number and the hand
    data = np.load(file)
    random.shuffle(data)

    if CURRENT_NUMBER in range(10):
        mask = np.any(data[:, :, 7] == CURRENT_NUMBER, axis=1)
        data = data[mask]

    if CURRENT_HAND in range(2):
        mask = np.any(data[:, :, 3] == CURRENT_HAND, axis=1)
        data = data[mask]
    print(data.shape)

    if CURRENT_FINGER in range(2):
        mask = np.any(data[:, :, 4] == CURRENT_FINGER, axis=1)
        data = data[mask]
    print(data.shape)

    ########################################
    # Ensure that data are balanced
    y_full = data[:, :, Y_FEATURE]
    counts = np.bincount(np.array(y_full[:, 0], dtype=int))
    count_zeros = int(counts[0])
    count_ones = int(counts[1])
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")

    min_count = min(count_zeros, count_ones)
    if min_count == 0:
        raise ValueError("One of the classes has no samples, data is imbalanced!")
    else:
        balanced_indices = np.hstack([
            np.random.choice(np.where(y_full[:, 0] == 0)[0], min_count, replace=False),
            np.random.choice(np.where(y_full[:, 0] == 1)[0], min_count, replace=False)
        ])
        np.random.shuffle(balanced_indices)
        data = data[balanced_indices]

    # Recalculate class counts after balancing
    counts_balanced = np.bincount(np.array(data[:, 0, Y_FEATURE], dtype=int))
    print(f"Balanced Number of 0s: {counts_balanced[0]}")
    print(f"Balanced Number of 1s: {counts_balanced[1]}")
    ########################################

    # split the data to train and validation
    split_index = int(0.8*len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    # split features and labels
    x_train = train_data[:, :, 0:2]
    y_train = train_data[:, :, Y_FEATURE]
    x_val = val_data[:, :, 0:2]
    y_val = val_data[:, :, Y_FEATURE]

    # adding velocity as feature
    time_train = train_data[:, :, 8]  # 8th index for time
    time_val = val_data[:, :, 8]
    velocity_train = calculate_velocity(x_train, time_train)
    velocity_val = calculate_velocity(x_val, time_val)
    x_train = np.concatenate((x_train, velocity_train), axis=-1)
    x_val = np.concatenate((x_val, velocity_val), axis=-1)

    # correcting the shape of labels
    y_train = np.squeeze(y_train[:, 0])
    y_val = np.squeeze(y_val[:, 0])

    # debugging code
    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_val))
    print(np.shape(y_val))

    # for sample in range(15):
    #     first_sample_X = x_train[sample, :, 0]
    #     first_sample_Y = x_train[sample, :, 1]
    #     plt.plot(first_sample_X, first_sample_Y,  marker='o', linestyle='-')
    #     plt.show()
    #
    #     first_sample_velocity = velocity_train[sample, :, 0]
    #     time_values = time_train[sample]
    #     plt.plot(time_values[:-1], first_sample_velocity[:-1], marker='o', linestyle='-', color='black')
    #     plt.show()
    #
    # assert 1==2

    # checking if classes ara balanced
    counts = np.bincount(np.array(y_train, dtype=int))
    count_zeros = counts[0]
    count_ones = counts[1]
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")

    # extracting the number of classes to dynamically
    num_classes = len(np.unique(y_train))
    print("num of classes ", num_classes)

    # building model
    model = Sequential()
    model.add(Input(shape=np.shape(x_train[0])))
    for i in range(NumberOfLSTM):
        model.add(LSTM(NeuronOfLSTMCells, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(LSTM(NeuronOfLSTMCells//2))
    model.add(Dropout(0.1))
    for i in range(NumberOfDense):
        model.add(Dense(NeuronOfDenseCells, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Metric to monitor
        patience=4,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore the weights of the model from the epoch with the best validation accuracy
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHES,
        batch_size=64,
        validation_data=(x_val, y_val),
        # callbacks=[early_stopping, accuracy_callback]
    )

    save_model(model, history, x_train.shape)
