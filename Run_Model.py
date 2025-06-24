import tensorflow as tf

# Path to the extracted model directory
model_path = "D:\\Thesis\\Model" 

# Load the model
model = tf.saved_model.load(model_path)

# Print available signatures (entry points for inference)
print("Model signatures:", model.signatures)

infer = model.signatures["default"]  # Get inference function
print("Input signature:", infer.structured_input_signature)
