import tensorflow as tf
from tensorflow import keras
import mlflow
import numpy as np
import io
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

L1, L2, L3 = 1.0, 1.5, 0.5  # link lengths

@tf.function
def forward_kinematics_tf(theta):
    theta1, theta2, theta3 = tf.unstack(theta, axis=1)
    
    x = L1 * tf.cos(theta1) * tf.sin(theta2) + L2 * tf.cos(theta1) * tf.sin(theta2 + theta3)
    y = L1 * tf.sin(theta1) * tf.sin(theta2) + L2 * tf.sin(theta1) * tf.sin(theta2 + theta3)
    z = L1 * tf.cos(theta2) + L2 * tf.cos(theta2 + theta3) + L3
    
    return tf.stack([x, y, z], axis=1)

def evaluate_model(model, test_inputs, test_outputs, input_mean, input_std, batch_size=2**16):
    test_inputs = tf.convert_to_tensor(test_inputs, dtype=tf.float32)
    input_mean = tf.convert_to_tensor(input_mean, dtype=tf.float32)
    input_std = tf.convert_to_tensor(input_std, dtype=tf.float32)
    predicted_angles_normalized = model.predict(test_inputs, batch_size=batch_size)
    predicted_angles = predicted_angles_normalized * (np.pi/2)
    true_xyz = test_inputs * input_std + input_mean
    predicted_xyz = forward_kinematics_tf(predicted_angles)
    errors = tf.norm(true_xyz - predicted_xyz, axis=1)
    return errors.numpy(), true_xyz.numpy(), predicted_xyz.numpy()

def plot_error_distribution(errors, title, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=400, alpha=0.5)
    plt.title(title)
    plt.xlabel('Error (Euclidean distance)')
    plt.ylabel('Frequency')
    plt.xlim([0, 0.5])
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_true_vs_predicted(true_xyz, predicted_xyz, title, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)
    
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        ax.scatter(true_xyz[:, i], predicted_xyz[:, i], alpha=0.1)
        ax.plot([true_xyz[:, i].min(), true_xyz[:, i].max()], [true_xyz[:, i].min(), true_xyz[:, i].max()], 'r--')
        ax.set_xlabel(f'True {coord}')
        ax.set_ylabel(f'Predicted {coord}')
        ax.set_title(f'{coord} Coordinate: True vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def load_and_preprocess_data(filename='robot_arm_dataset_10M.npz'):
    data = np.load(f'./Data/{filename}')
    inputs, outputs = data['inputs'], data['outputs']
    
    input_mean = np.mean(inputs, axis=0)
    input_std = np.std(inputs, axis=0)
    inputs_normalized = (inputs - input_mean) / input_std

    outputs_normalized = outputs / (np.pi/2)

    split_index = int(0.9 * len(inputs))
    train_inputs, test_inputs = inputs_normalized[:split_index], inputs_normalized[split_index:]
    train_outputs, test_outputs = outputs_normalized[:split_index], outputs_normalized[split_index:]

    return (train_inputs, train_outputs), (test_inputs, test_outputs), input_mean, input_std

class VerboseLoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        output = f"Epoch {epoch+1}/{self.params['epochs']} - "
        output += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(output)
        self.output.write(output + "\n")

    def get_output(self):
        return self.output.getvalue()

class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, 'value'):
            lr = lr.value()
        mlflow.log_metric("learning_rate", lr, step=epoch)
        
class CosineDecayWithWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super(CosineDecayWithWarmupCallback, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.current_step < self.warmup_steps:
            lr = self.initial_learning_rate * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.initial_learning_rate * (1 + np.cos(np.pi * progress))

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.current_step += 1