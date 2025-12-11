import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from keras.optimizers import Adam

#Utility class
import utils

# DATA
data_list = []
value_list = []

i=0
path = "dataset2/"
df = pd.read_csv(path+'driving_log.csv', header=None)
for row in df.itertuples(index=False):  # index=False to exclude the DataFrame index
    img_absolute_path = row[0]
    steering = row[3]

    #get image filename
    img_filename = os.path.basename(img_absolute_path)
    img_fullpath = path+"IMG/"+img_filename

    image = cv2.imread(img_fullpath)
    
    img = utils.preprocess(image)

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
# After train_test_split
print(f"\n=== DATASET STATISTICS ===")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_test)}")
print(f"Steering angle stats:")
print(f"  Mean: {y_train.mean():.4f}")
print(f"  Std: {y_train.std():.4f}")
print(f"  Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
print(f"Zero angles in training: {(np.abs(y_train) < 0.01).sum()} ({(np.abs(y_train) < 0.01).sum()/len(y_train)*100:.1f}%)")
print(f"Zero angles in validation: {(np.abs(y_test) < 0.01).sum()} ({(np.abs(y_test) < 0.01).sum()/len(y_test)*100:.1f}%)")

# Plot histogram as required in assignment (Figure 5)
plt.figure(figsize=(10, 4))
plt.hist(y_train, bins=50, edgecolor='black')
plt.title('Steering Angle Distribution')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.show()
#Data Augmentation
aug = ImageDataGenerator(
    width_shift_range=0.02,  # Reduced from 0.05
    zoom_range=0.05,         # Reduced from 0.1
    brightness_range=[0.9, 1.1],  # Reduced from [0.8, 1.2]
    fill_mode="nearest"
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

batch_size = 64  # Increased from 32 for more stable gradients
steps=len(X_train) // batch_size

# triplet loss
# MODEL
nn = Sequential([
        layers.Input(shape=(66,200,3)), # for input layer to avoid warning
        # layers.BatchNormalization(),
        layers.Conv2D(24, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(36, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(48, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Dropout(0.3),  # Add dropout before flattening
        layers.Flatten(),
        layers.Dense(1164, activation='relu'),
        layers.Dropout(0.5),  # Add dropout after first dense layer
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
])




nn.compile(optimizer=Adam(learning_rate=0.0001),  # Reduced from default 0.001
           loss='mse',
           metrics=['mae'])

H = nn.fit(flipped_flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), epochs=20, steps_per_epoch=steps) 
# H = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)
# EVALUATE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(H.history['loss'], label='train loss')
ax1.plot(H.history['val_loss'], label='validation loss')
ax1.set_title('Model Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(H.history['mae'], label='train MAE')
ax2.plot(H.history['val_mae'], label='validation MAE')
ax2.set_title('Model Mean Absolute Error (MAE)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()

plt.tight_layout()
plt.show()

#save model
nn.save("model.keras")


