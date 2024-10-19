import os
from datetime import datetime
import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.losses import SparseCategoricalCrossentropy
from train_NN import calculate_velocity
import itertools
import random


def save_model(model_, history_, c_hand, c_finger, shape_=None):
    final_val_accuracy = history_.history['val_accuracy'][-1] * 100
    current_time = datetime.now().strftime("%H%M%S")
    folder_path = "loops_run_1"

    c_finger = "index" if c_finger == 0 else "thumb"
    c_hand = "right"  if c_hand == 1 else "left"

    file_name = (
        f"{folder_path}/{final_val_accuracy:.2f}_{current_time}_"
        f"num{CURRENT_NUMBER}_{NumberOfDense}D_{NumberOfLSTM}L_"
        f"{NeuronOfDenseCells}DNeu_{NeuronOfLSTMCells}LNeu_"
        f"{EPOCHES}E_{c_finger}_{c_hand}_{shape_[0]}.keras"
    )
    model_.save(file_name)


if __name__ == '__main__':

    # (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number).
    Y_FEATURE = 2
    CURRENT_NUMER_list = [x for x in range(10)]
    NumberOfDense_list = [1,2,3]
    NumberOfLSTM_list = [1, 2, 3]
    NeuronOfDenseCells_list = [10, 20, 50]
    NeuronOfLSTMCells_list = [32, 64, 128]
    EPOCHES_list = [10, 20, 30]
    CURRENT_HAND_list = [0,1]
    CURRENT_FINGER_list = [0,1]

    random.seed(10)
    file = "processed_data_cubic_standard_with_time.npy"

    combinations = list(itertools.product(CURRENT_NUMER_list, NumberOfDense_list, NumberOfLSTM_list,
                                          NeuronOfDenseCells_list, NeuronOfLSTMCells_list, EPOCHES_list,
                                          CURRENT_HAND_list, CURRENT_FINGER_list))
    random.shuffle(combinations)
    sample_size = 100
    sampled_combinations = combinations[:sample_size]
    data = np.load(file)


    for (CURRENT_NUMBER, NumberOfDense, NumberOfLSTM, NeuronOfDenseCells, NeuronOfLSTMCells,
         EPOCHES,CURRENT_HAND, CURRENT_FINGER) in sampled_combinations:
        print(f"Current Number: {CURRENT_NUMBER}, "
              f"Number of Dense Layers: {NumberOfDense}, "
              f"Number of LSTM Layers: {NumberOfLSTM}, "
              f"Neurons in Dense Layer: {NeuronOfDenseCells}, "
              f"Neurons in LSTM Layer: {NeuronOfLSTMCells}, "
              f"Epochs: {EPOCHES}, "
              f"Current Finger: {CURRENT_FINGER}, "
              f"Current Hand: {CURRENT_HAND}")

        number_data = data
        if CURRENT_NUMBER in range(10):
            mask = np.any(data[:, :, 7] == CURRENT_NUMBER, axis=1)
            number_data = data[mask]
            print(f"number filtered {number_data.shape}")

        hand_data = data
        if CURRENT_HAND in range(2):
            mask = np.any(number_data[:, :, 3] == CURRENT_HAND, axis=1)
            hand_data = number_data[mask]
            print(f"hand filtered {hand_data.shape}")

        filtered_data = data
        if CURRENT_FINGER in range(2):
            mask = np.any(hand_data[:, :, 4] == CURRENT_FINGER, axis=1)
            filtered_data = hand_data[mask]
            print(f"total filtered {filtered_data.shape}")
        random.shuffle(filtered_data)

        ########################################
        # Ensure that data are balanced
        y_full = filtered_data[:, :, Y_FEATURE]
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
            filtered_data = filtered_data[balanced_indices]

        # Recalculate class counts after balancing
        counts_balanced = np.bincount(np.array(filtered_data[:, 0, Y_FEATURE], dtype=int))
        print(f"Balanced Number of 0s: {counts_balanced[0]}")
        print(f"Balanced Number of 1s: {counts_balanced[1]}")
        ########################################

        split_index = int(0.8 * len(filtered_data))
        train_data = filtered_data[:split_index]
        val_data = filtered_data[split_index:]

        x_train = train_data[:, :, 0:2]
        y_train = train_data[:, :, Y_FEATURE]
        x_val = val_data[:, :, 0:2]
        y_val = val_data[:, :, Y_FEATURE]

        time_train = train_data[:, :, 8]  # 8th index for time
        time_val = val_data[:, :, 8]
        velocity_train = calculate_velocity(x_train, time_train)
        velocity_val = calculate_velocity(x_val, time_val)
        x_train = np.concatenate((x_train, velocity_train), axis=-1)
        x_val = np.concatenate((x_val, velocity_val), axis=-1)

        y_train = np.squeeze(y_train[:, 0])
        y_val = np.squeeze(y_val[:, 0])

        counts = np.bincount(np.array(y_train, dtype=int))
        count_zeros = counts[0]
        count_ones = counts[1]

        num_classes = len(np.unique(y_train))

        model = Sequential()
        model.add(Input(shape=np.shape(x_train[0])))
        for i in range(NumberOfLSTM):
            model.add(LSTM(NeuronOfLSTMCells, return_sequences=True))
            model.add(Dropout(0.2))
        model.add(LSTM(NeuronOfLSTMCells // 2))
        model.add(Dropout(0.1))
        for i in range(NumberOfDense):
            model.add(Dense(NeuronOfDenseCells, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_accuracy',  # Metric to monitor
            patience=5,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore the weights of the model from the epoch with the best validation accuracy
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHES,
            batch_size=64,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping]
        )
        save_model(model, history, CURRENT_HAND, CURRENT_FINGER, x_train.shape)

