import os
os.environ["USE_TF"] = "0"
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(numerator, np.clip(denominator, 1e-6, None)), axis=0) * 100
    return smape


def save_mae_to_txt(mae_values, target_cols, filename="mae_results.txt", title="MAE Results"):
    with open(filename, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n")
        for name, score in zip(target_cols, mae_values):
            f.write(f"{name}: {score:.2f}\n")
    print(f"Saved MAE to {filename}")


def save_scatter_plots(y_true, y_pred, set_name="Test"):
    nutrients = ["Calories", "Fat", "Carbohydrates", "Protein"]
    folder = os.path.join("plots", set_name.lower())
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

        # Save with just the nutrient name
        safe_name = nutrient.lower().replace(" ", "_")
        filename = f"{safe_name}_scatter.png"
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

# === Load dataset ===
df = pd.read_csv("D:\Thesis\outputs\merged_training_dataset.csv")

# === Define target columns and segmentation features ===
target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]
ignore_cols = ["dish_id", "ingredients", "total_mass"] + target_cols
segmentation_cols = [col for col in df.columns if col not in ignore_cols]

# === Segmentations (raw pixel counts) ===
X_seg = df[segmentation_cols].values.astype(np.float32)

# === Encode ingredients using BERT ===
print("Encoding ingredients with BERT...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_bert = bert_model.encode(df["ingredients"].tolist(), show_progress_bar=True).astype(np.float32)

# === Concatenate both feature types ===
X = np.hstack((X_seg, X_bert))
y = df[target_cols].values.astype(np.float32)

# === Train/validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# === Build TensorFlow model ===
input_dim = X.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)  # 4 outputs: calories, fat, carb, protein
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Train the model ===
print("Training TensorFlow model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32, verbose=1)

# === Predict ===
y_pred = model.predict(X_val)
save_scatter_plots(y_val, y_pred, set_name="Train")

# === Evaluate ===
mae = mean_absolute_error(y_val, y_pred, multioutput='raw_values')

print("\n=== MAE per nutrient(train)===")
for name, score in zip(target_cols, mae):
    print(f"{name}: {score:.2f}")

save_mae_to_txt(mae, target_cols, filename="mae_segment.txt", title="MAE with Segmentation(train)")


# Load your test datasets
df_test_meta = pd.read_csv("D:\Thesis\outputs/test_dish_metadata_cleaned.csv")
df_test_seg = pd.read_csv("D:\Thesis\outputs\segmentation_test.csv")

# Merge on dish_id
df_test = pd.merge(df_test_meta, df_test_seg, on="dish_id")

# Define target and feature columns
target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]
segmentation_cols = [col for col in df_test.columns if col not in ["dish_id", "ingredients", "total_mass"] + target_cols]

# Prepare test features
X_seg_test = df_test[segmentation_cols].values.astype(np.float32)
X_bert_test = bert_model.encode(df_test["ingredients"].tolist(), show_progress_bar=True).astype(np.float32)
X_test = np.hstack((X_seg_test, X_bert_test))
y_test = df_test[target_cols].values.astype(np.float32)

# Predict using trained model
y_test_pred = model.predict(X_test)
save_scatter_plots(y_test, y_test_pred, set_name="Test")

# Evaluate
mae_test = mean_absolute_error(y_test, y_test_pred, multioutput='raw_values')

print("\n=== Test MAE per nutrient(test) ===")
for name, score in zip(target_cols, mae_test):
    print(f"{name}: {score:.2f}")

save_mae_to_txt(mae_test, target_cols, filename="mae_segment.txt", title="MAE with Segmentation(test)")


# Save predicted values to CSV
pred_df = pd.DataFrame(y_test_pred, columns=target_cols)
pred_df.insert(0, "dish_id", df_test["dish_id"])
pred_df.to_csv("predicted_test_values.csv", index=False)
print("Saved predicted test values to predicted_test_values.csv")

# Save groundtruth values to CSV
truth_df = pd.DataFrame(y_test, columns=target_cols)
truth_df.insert(0, "dish_id", df_test["dish_id"])
truth_df.to_csv("groundtruth_test_values.csv", index=False)
print("Saved groundtruth test values to groundtruth_test_values.csv")


# After predicting
smape_train = calculate_smape(y_val, y_pred)
smape_test = calculate_smape(y_test, y_test_pred)

print("\n=== SMAPE per nutrient (train) ===")
for name, score in zip(target_cols, smape_train):
    print(f"{name}: {score:.2f}%")

print("\n=== SMAPE per nutrient (test) ===")
for name, score in zip(target_cols, smape_test):
    print(f"{name}: {score:.2f}%")


save_mae_to_txt(smape_test, target_cols, filename="smape_test.txt", title="SMAPE (Test Set)")
save_mae_to_txt(smape_train, target_cols, filename="smape_train.txt", title="SMAPE (Train Set)")




# === SECOND EXPERIMENT: BERT-only model (no segmentation) ===
print("\n\n==== Running NO SEGMENTATION baseline (BERT only) ====\n")

# Only BERT features
X_only_bert = X_bert
X_bert_train, X_bert_val, y_train_bert, y_val_bert = train_test_split(X_only_bert, y, test_size=0.1, random_state=42)

# Build BERT-only model
model_bert = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_only_bert.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)
])
model_bert.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
print("Training BERT-only model...")
model_bert.fit(X_bert_train, y_train_bert, validation_data=(X_bert_val, y_val_bert), epochs=100, batch_size=32, verbose=1)

# Predict on validation
y_pred_bert_val = model_bert.predict(X_bert_val)
save_scatter_plots(y_val_bert, y_pred_bert_val, set_name="no_segment/train")

# Evaluate
mae_bert_train = mean_absolute_error(y_val_bert, y_pred_bert_val, multioutput='raw_values')
print("\n=== MAE per nutrient (no segmentation, train) ===")
for name, score in zip(target_cols, mae_bert_train):
    print(f"{name}: {score:.2f}")

# --- Test set ---
X_bert_test_only = X_bert_test
y_pred_bert_test = model_bert.predict(X_bert_test_only)
save_scatter_plots(y_test, y_pred_bert_test, set_name="no_segment/test")

mae_bert_test = mean_absolute_error(y_test, y_pred_bert_test, multioutput='raw_values')
print("\n=== MAE per nutrient (no segmentation, test) ===")
for name, score in zip(target_cols, mae_bert_test):
    print(f"{name}: {score:.2f}")

save_mae_to_txt(mae_bert_train, target_cols, filename="mae_no_segment_train.txt", title="MAE (No Segmentation, Train)")
save_mae_to_txt(mae_bert_test, target_cols, filename="mae_no_segment_test.txt", title="MAE (No Segmentation, Test)")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss over Epochs")
plt.show()