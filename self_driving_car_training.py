import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# DATA
data_list = []
value_list = []

img_size = (200,66)
i=0
path = "dataset/"
df = pd.read_csv(path+'driving_log.csv', header=None)
for row in df.itertuples(index=False):  # index=False to exclude the DataFrame index
    img_absolute_path = row[0]
    steering = row[3]

    #get image filename
    img_filename = img_absolute_path.split('\\')[-1]
    img_fullpath = path+"IMG/"+img_filename

    image = cv2.imread(img_fullpath)
    dimensions = image.shape
    image_w = dimensions[1]

    #crop image
    x_start = 0
    y_start = 60
    width = image_w
    height = 135 - 60
    cropped_image = image[y_start : y_start + height, x_start : x_start + width]

    #convert to YUV color
    img_yuv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2YUV)

    #resize image
    img = cv2.resize(img_yuv, img_size)

    #apply Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    #Normalize values
    image_f = img/255

    # #Flatten
    # image_f = image_f.flatten()

    #store to lists
    data_list.append(img)
    value_list.append(steering)

    if i%200 == 0:
        print(f'[INFO] {i} images read!')
        if i > 1000:
            break

    i += 1

cv2.destroyAllWindows()

X = np.array(data_list)
y = np.array(value_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Data Augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
)

# triplet loss
# MODEL
nn = Sequential([
        layers.Conv2D(24, (5,5), activation='relu', input_shape=(66,200,3)),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(36, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(48, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        # layers.Conv2D(64, (3,3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(1164, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
])



nn.compile(optimizer='adam',
           loss='MSE',
           metrics=['MAE'])

H = nn.fit(aug.flow(X_train, y_train), validation_data=(X_test, y_test), epochs=10, batch_size=32)


# EVALUATE
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='validation loss')
plt.plot(H.history['MAE'], label='MAE')
plt.plot(H.history['val_MAE'], label='validation MAE')
plt.legend()

plt.show()

