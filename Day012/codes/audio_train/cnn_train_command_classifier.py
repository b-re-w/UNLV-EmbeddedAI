import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, roc_auc_score, confusion_matrix, classification_report)
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = "cmd_dataset"
SAMPLE_RATE = 16000
DURATION = 1  # Seconds
N_MFCC = 13  # Number of MFCC coefficients
HOP_LENGTH = 512
N_FFT = 2048
BATCH_SIZE = 32
EPOCHS = 20

# Valid classes (folders) to look for
# We explicitly handle _background_noise_ as a distinct class "silence" or keep name
TARGET_CLASSES = ["up", "down", "on", "off", "left", "right", "tello", "_background_noise_"]


# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_mfcc(y, sr):
    # Ensure audio is exactly 1 second
    if len(y) < sr:
        # Pad with zeros if too short
        y = np.pad(y, (0, sr - len(y)), 'constant')
    else:
        # Crop if too long
        y = y[:sr]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc.T  # Transpose to (Time, Feats)


def load_dataset(path):
    X = []
    y = []

    print("Loading dataset...")

    if not os.path.exists(path):
        print(f"Error: Dataset folder '{path}' not found.")
        return np.array([]), np.array([])

    for label in os.listdir(path):
        class_path = os.path.join(path, label)

        # Skip if not a folder or not in our target list (optional filter)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing '{label}'...")

        for file in os.listdir(class_path):
            if not file.endswith(".wav"): continue

            file_path = os.path.join(class_path, file)

            try:
                # Load audio
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # SPECIAL HANDLING: _background_noise_
                # These are usually long files (mins). We slice them into 1s chunks.
                if label == "_background_noise_":
                    # Slice into 1-second non-overlapping chunks
                    num_chunks = int(len(signal) / SAMPLE_RATE)
                    for i in range(num_chunks):
                        chunk = signal[i * SAMPLE_RATE: (i + 1) * SAMPLE_RATE]
                        mfcc = extract_mfcc(chunk, SAMPLE_RATE)
                        X.append(mfcc)
                        y.append(label)
                else:
                    # Normal commands (usually ~1s)
                    mfcc = extract_mfcc(signal, SAMPLE_RATE)
                    X.append(mfcc)
                    y.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return np.array(X), np.array(y)


# Load Data
X, y = load_dataset(DATASET_PATH)

if len(X) == 0:
    print("No data loaded. Please check folder structure.")
    exit()

# Add Channel Dimension for CNN (Batch, Time, MFCC, 1)
X = X[..., np.newaxis]

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
classes = le.classes_

print(f"\nData Shape: {X.shape}")
print(f"Classes found: {classes}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42,
                                                    stratify=y_encoded)


# ==========================================
# 3. DEFINE CNN MODEL
# ==========================================
def create_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Conv Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Conv Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten & Dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_cnn(X_train.shape[1:], len(classes))
model.summary()

# ==========================================
# 4. TRAINING
# ==========================================
print("\nStarting Training...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

# Save Model
model.save("audio_classifier.h5")
print("\nModel saved to audio_classifier.h5")

# ==========================================
# 5. PERFORMANCE ESTIMATION
# ==========================================
print("\n--- Model Evaluation ---")

# Predict Probabilities and Classes
y_pred_prob = model.predict(X_test)
y_pred_class = np.argmax(y_pred_prob, axis=1)
y_true_class = np.argmax(y_test, axis=1)

# 1. Accuracy
acc = accuracy_score(y_true_class, y_pred_class)

# 2. Precision, Recall, F1 (Macro Average handles class imbalance better)
precision = precision_score(y_true_class, y_pred_class, average='macro')
recall = recall_score(y_true_class, y_pred_class, average='macro')
f1 = f1_score(y_true_class, y_pred_class, average='macro')

# 3. Log Loss (Categorical Crossentropy on Test Set)
ll = log_loss(y_test, y_pred_prob)

# 4. ROC-AUC (One-vs-Rest for Multi-class)
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
except ValueError:
    roc_auc = 0.0
    print("Warning: ROC-AUC requires all classes to be present in test set.")

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Log Loss:  {ll:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# 5. Confusion Matrix Visualization
cm = confusion_matrix(y_true_class, y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 6. Detailed Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_true_class, y_pred_class, target_names=classes))