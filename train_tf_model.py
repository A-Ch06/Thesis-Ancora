import os
os.environ["USE_TF"] = "0"
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ====== CONFIG ======
TEXT_OUT = "D:/Thesis/mae_files"
RESULTS_OUT = "D:/Thesis/Results"
MODEL_PATH = "D:/Thesis/models/nutrient_predictor_segmentation"
PLOTS_PATH = "D:/Thesis/plots"
BEST_MODEL_PATH = "D:/Thesis/models/best_nutrient_predictor.keras"

os.makedirs(TEXT_OUT, exist_ok=True)
os.makedirs(RESULTS_OUT, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# ====== UTILITY FUNCTIONS ======
def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(numerator, np.clip(denominator, 1e-6, None)), axis=0) * 100
    return smape

def save_mae_to_txt(mae_values, target_cols, filename, title):
    full_path = os.path.join(TEXT_OUT, filename)
    with open(full_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n")
        for name, score in zip(target_cols, mae_values):
            f.write(f"{name}: {score:.2f}\n")
    print(f"Saved: {full_path}")

def save_scatter_plots(y_true, y_pred, set_name="Test"):
    nutrients = ["Calories", "Fat", "Carbohydrates", "Protein"]
    folder = os.path.join(PLOTS_PATH, set_name.lower())
    os.makedirs(folder, exist_ok=True)
    for i, nutrient in enumerate(nutrients):
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, edgecolor='k')
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
        plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal")
        plt.xlabel(f"Actual {nutrient}")
        plt.ylabel(f"Predicted {nutrient}")
        plt.title(f"{set_name} Set: {nutrient}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename = f"{nutrient.lower().replace(' ', '_')}_scatter.png"
        plt.savefig(os.path.join(folder, filename), dpi=300)
        plt.close()
        print(f"Saved: {os.path.join(folder, filename)}")

def save_loss_curve(history):
    plt.figure(figsize=(6, 5))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    full_path = os.path.join(PLOTS_PATH, "loss_over_epochs.png")
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved: {full_path}")

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# === Load dataset ===
df = pd.read_csv("D:/Thesis/outputs/merged_training_dataset.csv")
target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]
ignore_cols = ["dish_id", "ingredients", "total_mass"] + target_cols
segmentation_cols = [col for col in df.columns if col not in ignore_cols]

# === Feature preparation ===
X_seg = df[segmentation_cols].values.astype(np.float32)
print("Encoding ingredients with BERT...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_bert = bert_model.encode(df["ingredients"].tolist(), show_progress_bar=True).astype(np.float32)

X = np.hstack((X_seg, X_bert))
y = df[target_cols].values.astype(np.float32)

# === Train/validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED)

# === Build TensorFlow model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Train the model ===
print("Training TensorFlow model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1, callbacks=[checkpoint_cb])

# === Evaluate on validation ===
y_pred_val = model.predict(X_val)
save_scatter_plots(y_val, y_pred_val, set_name="Train")
mae_val = mean_absolute_error(y_val, y_pred_val, multioutput='raw_values')
save_mae_to_txt(mae_val, target_cols, filename="mae_train.txt", title="MAE - Train Set")
smape_train = calculate_smape(y_val, y_pred_val)
save_mae_to_txt(smape_train, target_cols, filename="smape_train.txt", title="SMAPE - Train Set")

# === Save model ===
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# === Load test data and predict ===
df_meta = pd.read_csv("D:/Thesis/outputs/test_dish_metadata_cleaned.csv")
df_seg = pd.read_csv("D:/Thesis/outputs/segmentation_test.csv")
df_test = pd.merge(df_meta, df_seg, on="dish_id")

X_seg_test = df_test[segmentation_cols].values.astype(np.float32)
X_bert_test = bert_model.encode(df_test["ingredients"].tolist(), show_progress_bar=True).astype(np.float32)
X_test = np.hstack((X_seg_test, X_bert_test))
y_test = df_test[target_cols].values.astype(np.float32)

# === Load saved model and evaluate ===
print("Loading saved model...")
model = tf.keras.models.load_model(BEST_MODEL_PATH)
y_pred_test = model.predict(X_test)

save_scatter_plots(y_test, y_pred_test, set_name="Test")
mae_test = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
save_mae_to_txt(mae_test, target_cols, filename="mae_test.txt", title="MAE - Test Set")
smape_test = calculate_smape(y_test, y_pred_test)
save_mae_to_txt(smape_test, target_cols, filename="smape_test.txt", title="SMAPE - Test Set")

# === Save predictions and ground truth ===
pred_df = pd.DataFrame(y_pred_test, columns=target_cols)
pred_df.insert(0, "dish_id", df_test["dish_id"])
pred_df.to_csv(os.path.join(RESULTS_OUT, "predicted_test_values.csv"), index=False)

truth_df = pd.DataFrame(y_test, columns=target_cols)
truth_df.insert(0, "dish_id", df_test["dish_id"])
truth_df.to_csv(os.path.join(RESULTS_OUT, "groundtruth_test_values.csv"), index=False)

print("Predictions and ground truth saved to Results folder.")
save_loss_curve(history)
 