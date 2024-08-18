import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

####################
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Initialize arrays for x, y, and gender
x_data = []
y_data = []
genders = []

# Balancing the dataset: 500 males and 500 females
num_males = num_samples // 2
num_females = num_samples // 2

while len(genders) < num_samples:
    # Generate random x and y sequences with 5 time steps
    x_seq = np.random.randint(1, 50, 5).tolist()
    y_seq = np.random.randint(1, 50, 5).tolist()

    # Calculate the sum condition
    x_sum = sum(x_seq)
    y_sum = sum(y_seq)
    condition = x_sum*2-5 > y_sum

    if condition == True and num_males > 0:
        # Male condition
        genders.append('male')
        num_males -= 1
    elif condition ==False and num_females > 0:
        # Female condition
        genders.append('female')
        num_females -= 1
    else:
        continue

    # Append x and y sequences
    x_data.append(x_seq)
    y_data.append(y_seq)

# Convert lists to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)
genders = np.array(genders)

# Print the shape of the arrays to confirm
print(f"x_data shape: {x_data.shape}")
print(f"y_data shape: {y_data.shape}")
print(f"genders shape: {genders.shape}")
num_males = np.sum(genders == 'male')
print(num_males)
# Example to visualize a few samples
for i in range(5):
    print(f"Sample {i + 1}: x={x_data[i]}, y={y_data[i]}, gender={genders[i]}")


x_data_normalized = x_data #/ (np.max(x_data)-(np.min(x_data)))
y_data_normalized = y_data #/ (np.max(y_data)-(np.min(y_data)))

for i in range(5):
    print(f"Sample {i + 1}: x={x_data[i]}, y={y_data[i]}, gender={genders[i]}")

######################

# x_data = [[2, 3, 5], [2, 5, 2, 1], [23, 2, 21, 41]]
# y_data = [[2, 1, 4], [4, 2, 1, 5], [43, 24, 14, 53]]
# genders = ['male', 'female', 'female']

sequences = []

for x_seq, y_seq in zip(x_data_normalized, y_data_normalized):
    seq = [[x, y] for x, y in zip(x_seq, y_seq)]
    sequences.append(seq)

max_seq_len = max(len(seq) for seq in sequences)

sequences =[[[0,0]]*(max_seq_len-len(seq))+seq for seq in sequences]
print(np.shape(sequences))
print(genders[0])
encoder = OneHotEncoder(sparse_output=False)
genders_encoded = encoder.fit_transform(np.array(genders).reshape(-1, 1))
print(genders_encoded[0])
sequence_input_shape = (max_seq_len, 2)
gender_input_shape = (genders_encoded.shape[1],)


model = Sequential()
model.add(Input(shape=sequence_input_shape))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(8))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(gender_input_shape[0], activation='softmax'))  # if you want to predict genders as an output
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert sequences to numpy array
sequences = np.array(sequences)

X_train, X_val, y_train, y_val = train_test_split(
    sequences,
    genders_encoded,
    test_size=0.1,  # 10% of the data for validation
    random_state=42  # Seed for reproducibility
)

# Train the model

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=50,
    validation_data=(X_val, y_val)  # Pass the validation data
)

x_sample_1 = [[100, 3], [3, 4], [2, 3], [1, 3], [2, 4]]
x_sample_2 = [[1, 8], [2, 9], [2, 6], [1, 4], [0, 0]]

# Convert to NumPy arrays and add batch dimension
x_data = np.array([x_sample_1, x_sample_2])

# Predict using the model
# Predict using the model
predictions = model.predict(x_data)
print(predictions)
