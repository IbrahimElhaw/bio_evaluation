import numpy as np

from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.losses import SparseCategoricalCrossentropy
from train_NN import save_model, calculate_velocity
import itertools
import random



# (0:X , 1:Y, 2:gender, 3:hand, 4:finger, 5:language, 6:age, 7:number).
Y_FEATURE = 2
CURRENT_NUMER_list = [x for x in range(10)]
NumberOfDense_list = [1,2,3]
NumberOfLSTM_list = [1, 2, 3]  # should be even %2=0
NeuronOfDenseCells_list = [10, 20, 50]
NeuronOfLSTMCells_list = [32, 64, 128]
EPOCHES_list = [10, 20, 30]
random.seed(10)
file = "processed_data_cubic_standard_with_time.npy"

combinations = list(itertools.product(CURRENT_NUMER_list, NumberOfDense_list, NumberOfLSTM_list,
                                      NeuronOfDenseCells_list, NeuronOfLSTMCells_list, EPOCHES_list))
random.shuffle(combinations)
sample_size = 100
sampled_combinations = combinations[:sample_size]
data = np.load(file)
for (CURRENT_NUMER, NumberOfDense, NumberOfLSTM, NeuronOfDenseCells, NeuronOfLSTMCells, EPOCHES) in sampled_combinations:
    print((CURRENT_NUMER, NumberOfDense, NumberOfLSTM, NeuronOfDenseCells, NeuronOfLSTMCells, EPOCHES))
    mask = np.any(data[:, :, 7] == CURRENT_NUMER, axis=1)
    one_number_data = data[mask]
    random.shuffle(one_number_data)

    print(one_number_data.shape)

    split_index = int(0.8 * len(one_number_data))
    train_data = one_number_data[:split_index]
    val_data = one_number_data[split_index:]

    x_train = train_data[:, :, 0:2]
    y_train = train_data[:, :, Y_FEATURE]
    x_val = val_data[:, :, 0:2]
    y_val = val_data[:, :, Y_FEATURE]

    angles = np.arctan2(x_train[:, :, 1], x_train[:, :, 0])
    angles = np.expand_dims(angles, axis=-1)
    x_train = np.concatenate((x_train, angles), axis=-1)
    angles = np.arctan2(x_val[:, :, 1], x_val[:, :, 0])
    angles = np.expand_dims(angles, axis=-1)
    x_val = np.concatenate((x_val, angles), axis=-1)

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
        patience=4,  # Number of epochs with no improvement after which training will be stopped
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
    save_model(model, history)
