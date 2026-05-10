import os
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── 1. Download dataset ────────────────────────────────────────────────────────
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
print("Downloaded to:", path)

# ── 2. Locate the image root (walk until we find class subdirectories) ─────────
def find_image_root(base_path):
    """Return the first directory whose children are all subdirectories (class folders)."""
    for root, dirs, files in os.walk(base_path):
        # Ignore hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if dirs and not files:          # folder contains only subdirs → likely class root
            return root
    return base_path                    # fallback: use the download root as-is

data_dir = find_image_root(path)
print("Using data directory:", data_dir)

# ── 3. Image preprocessing ────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Classes found: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples} | Validation samples: {val_gen.samples}")

# ── 4. Build model ────────────────────────────────────────────────────────────
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 5. Callbacks ──────────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

# ── 6. Stage 1 — train top layers only ───────────────────────────────────────
print("\n=== Stage 1: Training top layers ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

# ── 7. Stage 2 — fine-tune last 50 layers of base ────────────────────────────
print("\n=== Stage 2: Fine-tuning last 50 layers ===")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# ── 8. Save model ─────────────────────────────────────────────────────────────
model.save('waste_sorter_optimized.keras')
print("Model saved to waste_sorter_optimized.keras")
