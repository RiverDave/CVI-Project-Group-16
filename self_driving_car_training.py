import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

#Utility class
import utils

"""
Enable mixed precision for performance, essentially:
- Basically, All layers will use f32 to store data, BUT:
    - The computations will be done in float16 where possible, this is given the nature of GPUs (faster with float16)
"""
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# DATA
data_list = []
value_list = []

i=0
path = "dataset/"
df = pd.read_csv(path+'driving_log.csv', header=None)
for row in df.itertuples(index=False):  # index=False to exclude the DataFrame index
    # Column 0: center, Column 1: left, Column 2: right
    center_path = row[0]
    left_path = row[1]
    right_path = row[2]
    steering = row[3]

    # Process all 3 cameras
    cameras = [
        (center_path, steering),
    ]

    for img_path, adjusted_steering in cameras:
        # Get image filename (handle both Windows and Unix paths)
        img_filename = img_path.strip().replace('\\', '/').split('/')[-1]
        img_fullpath = path + "IMG/" + img_filename

        image = cv2.imread(img_fullpath)
        if image is None:
            continue  # Skip if image not found
        
        img = utils.preprocess(image)

        # Store to lists
        data_list.append(img)
        value_list.append(adjusted_steering)

    if i % 200 == 0:
        print(f'[INFO] {i} rows processed ({len(data_list)} images)!')

    i += 1

print(f'[INFO] Total images loaded: {len(data_list)}')

X = np.array(data_list)
y = np.array(value_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create tf.data pipeline (GPU-optimized)
batch_size = 64  # Larger batch size for GPU

def augment_fn(image, steering):
    """GPU-compatible augmentation with shadow and translation."""
    
    if tf.random.uniform(()) < 0.5:
        image = tf.image.flip_left_right(image)
        steering = -steering
    
    image = tf.image.random_brightness(image, 0.3)
    
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, steering

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(len(X_train)) \
    .map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

steps = len(X_train) // batch_size

# The following Architecture is based on the NVIDIA's paper: End to End Learning for Self-Driving Cars
nn = Sequential([
        layers.Input(shape=(66,200,3)),
        layers.BatchNormalization(),
        layers.Conv2D(24, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(36, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(48, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(1164, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, dtype='float32')
])


nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower LR for stability
           loss='MSE',
           metrics=['MAE'])

# Add early stopping and learning rate reduction
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

H = nn.fit(train_dataset, validation_data=val_dataset, epochs=50, 
           steps_per_epoch=steps, callbacks=[early_stop, reduce_lr]) 

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


