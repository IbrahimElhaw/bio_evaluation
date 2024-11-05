import matplotlib.pyplot as plt
import numpy as np
import glob
import os

directory = r"C:\Users\himaa\PycharmProjects\Numerical3\synthetic\represented"
file_pattern = os.path.join(directory, "1_6_thumb_3_male_right_27*.npz")
file_list = glob.glob(file_pattern)

def get_all_files(folder_path):
    # List all files in the specified directory only
    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename))
    ]

    return file_paths


files1 = get_all_files(r"C:\Users\himaa\PycharmProjects\Numerical3\synthetic\original")
files2 = get_all_files(r"C:\Users\himaa\PycharmProjects\Numerical3\synthetic\represented")
files3 = get_all_files(r"C:\Users\himaa\PycharmProjects\Numerical3\synthetic\manipulated")
files4 = get_all_files(r"C:\Users\himaa\PycharmProjects\Numerical3\synthetic\manipulated_2_modes")

for file1, file2, file3, file4 in zip(files1, files2, files3, files4):
    print(file2.split("_")[-1])


    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)
    data3 = np.load(file3, allow_pickle=True)
    data4 = np.load(file4, allow_pickle=True)
    for i in range(len(data1["X1"])):
        plt.plot(data1["X1"][i], data1["Y1"][i])
        plt.plot(data2["X1"][i], data2["Y1"][i])
        plt.plot(data3["X1"][i], data3["Y1"][i])
        # plt.plot(data4["X1"][i], data4["Y1"][i])
    plt.axis("equal")
    plt.show()





