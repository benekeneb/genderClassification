import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

rootdir = 'utkface'
max_iteration = 100000

path, dirs, files = next(os.walk(rootdir))
image_count = len(files)

if max_iteration < image_count:
    image_count = max_iteration

gender_array = np.zeros((image_count+1))
age_array = np.zeros((image_count+1))

i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)

        # Load Image to NP array
        if not file.endswith('.jpg'):
            continue

        # Label 
        filename_array = file.split("_")
        age = filename_array[0]
        gender = filename_array[1]
        race = filename_array[2]
        # print("age: " + str(age) + ", gender: " + str(gender) + ", race: " + str(race))
        # print("")

        # Append to Dataset Matrix
        gender_array[i] = gender
        age_array[i] = age

        i+=1
        if i > max_iteration:
            break


fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=False)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(gender_array, ec='black')
axs[1].hist(age_array, bins=range(60), ec='black')
plt.savefig('utkface_histograms.png')
plt.show()