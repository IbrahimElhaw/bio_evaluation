import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler

TEST_SAMPLE = 11


def interpolate_sample(sample_, n_points):
    # Interpolate numerical columns
    x_values = np.arange(len(sample_))
    x_new = np.linspace(0, len(sample_) - 1, n_points)

    # Cubic spline interpolation for 'X' and 'Y'
    cs_x = CubicSpline(x_values, sample_['X'])
    cs_y = CubicSpline(x_values, sample_['Y'])

    interpolated_X = cs_x(x_new)
    interpolated_Y = cs_y(x_new)

    # Creating a new DataFrame for the interpolated data
    interpolated_df = pd.DataFrame({
        'X': interpolated_X,
        'Y': interpolated_Y
    })

    # Extend categorical columns
    categorical_columns = ['hand', 'gender', 'language', 'finger', 'number', 'age']
    for column in categorical_columns:
        value = sample_[column].iloc[0]
        interpolated_df[column] = [value] * n_points

    return interpolated_df


df_subject = pd.read_csv('subject.csv', sep=',')
df_glyph = pd.read_csv('glyph.csv', sep=",")
df_strokes = pd.read_csv('stroke.csv', sep=",")
df_touches = pd.read_csv('touch.csv', sep=",")


def rename_columns(df, prefix):
    df.columns = [f'{prefix}_{col}' for col in df.columns]
    return df


# Renaming the columns
df_subject = rename_columns(df_subject, 'subject')
df_glyph = rename_columns(df_glyph, 'glyph')
df_strokes = rename_columns(df_strokes, 'stroke')
df_touches = rename_columns(df_touches, 'touch')

main_df = pd.merge(df_touches, df_strokes, left_on='touch_ZSTROKE', right_on='stroke_Z_PK', how='inner')
main_df = pd.merge(main_df, df_glyph, left_on='stroke_ZGLYPH', right_on='glyph_Z_PK', how='inner')
main_df = pd.merge(main_df, df_subject, left_on='glyph_ZSUBJECT', right_on='subject_Z_PK', how='inner')

relevant_data = ["glyph_Z_PK",
                 "stroke_Z_PK",
                 "touch_ZTIMESTAMP",
                 "touch_ZX",
                 "touch_ZY",
                 "subject_ZHANDEDNESS",
                 "subject_ZSEX",
                 "subject_ZNATIVELANGUAGE",
                 "glyph_ZFINGER",
                 "glyph_ZCHARACTER",
                 "subject_ZAGE"]

main_df = main_df[relevant_data]
main_df.set_index("glyph_Z_PK", inplace=True)
main_df.sort_values(by=["glyph_Z_PK", "stroke_Z_PK", "touch_ZTIMESTAMP"], inplace=True)

main_df.drop(columns=["stroke_Z_PK", "touch_ZTIMESTAMP"], inplace=True)
main_df.columns = ["X", "Y", "hand", "gender", "language", "finger", "number", "age"]

main_df['age'] = main_df['age'].apply(lambda x: 0 if x < 16 else 1)
main_df['hand'] = main_df['hand'].apply(lambda x: 1 if x == 'right' else 0)
main_df['gender'] = main_df['gender'].apply(lambda x: 1 if x == 'male' else 0)
main_df['language'] = main_df['language'].apply(lambda x: 0 if x == 'IE' else 1)
main_df['finger'] = main_df['finger'].apply(lambda x: 0 if x == 'index' else 1)

main_df = main_df.drop_duplicates()

indexes = main_df.index.unique()
MAX = max(main_df.groupby(main_df.index).size())

prediction_data = []
print("entering loop")
it = 0
for index in indexes:
    sample_data = []
    sample = main_df.loc[index]
    interpolated_sample = interpolate_sample(sample, MAX)
    for i, row in interpolated_sample.iterrows():
        point_x = row["X"]
        point_y = -row["Y"]  # flip around x-axis to be readable
        point_gender = row["gender"]
        point_age = row["age"]
        point_hand = row["hand"]
        point_finger = row["finger"]
        point_language = row["language"]
        point_number = row["number"]
        point = [point_x, point_y, point_gender, point_hand, point_finger, point_language, point_age, point_number]
        sample_data.append(point)
    prediction_data.append(sample_data)
    # if it > 30:
    #     break
    # it += 1
print("exiting loop")

prediction_data = np.array(prediction_data)

scaler = MinMaxScaler(feature_range=(0, 1))

x_flatten = prediction_data[:, :, 0].flatten().reshape(-1, 1)
y_flatten = prediction_data[:, :, 1].flatten().reshape(-1, 1)

# Fit the scaler to the data and transform it
x_scaled = scaler.fit_transform(x_flatten).reshape(prediction_data.shape[0], prediction_data.shape[1])
y_scaled = scaler.fit_transform(y_flatten).reshape(prediction_data.shape[0], prediction_data.shape[1])

# Assign the normalized values back to the prediction_data array
prediction_data[:, :, 0] = x_scaled
prediction_data[:, :, 1] = y_scaled

# for sample in prediction_data[:, :, 0:2]:
#     x = sample[:,0]
#     y = sample[:,1]
#     plt.plot(x, y, marker="o", zorder=1)
#     plt.scatter(x[0], y[0], color="purple", zorder=2)
#     plt.show()
print("saving file")

np.save('processed_data.npy', prediction_data)
