import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# import joblib

#Utility class
import utils

# DATA
data_list = []
value_list = []

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
    
    img = utils.preprocess(image)

    # #Flatten
    # image_f = image_f.flatten()

    #store to lists
    data_list.append(img)
    value_list.append(steering)

    if i%200 == 0:
        print(f'[INFO] {i} images read!')
        # if i > 1000:
        #     break

    i += 1

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

def flipped_flow(X, y, batch_size, flip_prob=0.5):
    """Yield augmented batches; optionally flip image and negate steering."""
    gen = aug.flow(X, y, batch_size=batch_size, shuffle=True)
    while True:
        Xb, yb = next(gen)
        for i in range(len(Xb)):
            if np.random.rand() < flip_prob:
                Xb[i] = cv2.flip(Xb[i], 1)  # horizontal flip
                yb[i] = -yb[i]             # reverse steering angle
        yield Xb, yb

batch_size = 32
steps=len(X_train) // batch_size

# triplet loss
# MODEL
nn = Sequential([
        layers.Input(shape=(66,200,3)), # for input layer to avoid warning
        layers.BatchNormalization(),
        layers.Conv2D(24, (5,5), strides=(2,2), activation='relu', input_shape=(66,200,3)),
        layers.Conv2D(36, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(48, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        # layers.Dropout(0.5),
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

H = nn.fit(flipped_flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), epochs=30, steps_per_epoch=steps) 

# EVALUATE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(H.history['loss'], label='train loss')
ax1.plot(H.history['val_loss'], label='validation loss')
ax1.set_title('Model Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(H.history['MAE'], label='train MAE')
ax2.plot(H.history['val_MAE'], label='validation MAE')
ax2.set_title('Model Mean Absolute Error (MAE)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()

plt.tight_layout()
plt.show()

#save model
nn.save("model.keras")


