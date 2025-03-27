import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Hyperparameters
IMG_SIZE = (224, 224)  # Standard size for MobileNetV2
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.00098

DATASET_DIR = "Data"

def split_dataset(dataset_dir, train_dir, val_dir, split_ratio=0.8):
    """Splits dataset into train and validation sets while handling the nested 'images' folder."""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category, "images")  # Access "images" folder inside each class
        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        images = os.listdir(category_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create category directories in train/val folders
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        # Move images
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))

        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

# Split dataset
split_dataset(DATASET_DIR, train_dir, val_dir)

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Load MobileNetV2 
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Create model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save("keras_model.h5")
