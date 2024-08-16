import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import mean_absolute_error

def forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3):
    x = l1 * np.cos(theta1) * np.sin(theta2) + l2 * np.cos(theta1) * np.sin(theta2 + theta3)
    y = l1 * np.sin(theta1) * np.sin(theta2) + l2 * np.sin(theta1) * np.sin(theta2 + theta3)
    z = l1 * np.cos(theta2) + l2 * np.cos(theta2 + theta3) + l3
    return x, y, z

def generate_dataset(num_samples=10_000_000):
    l1, l2, l3 = 1.0, 1.5, 0.5  # link lengths
    theta1 = np.random.uniform(-np.pi/2, np.pi/2, num_samples)
    theta2 = np.random.uniform(-np.pi/2, np.pi/2, num_samples)
    theta3 = np.random.uniform(-np.pi/2, np.pi/2, num_samples)
    
    inputs = np.array([forward_kinematics_3dof(t1, t2, t3, l1, l2, l3) for t1, t2, t3 in zip(theta1, theta2, theta3)])
    outputs = np.column_stack((theta1, theta2, theta3))
    
    return inputs, outputs

def save_dataset(inputs, outputs, filename='robot_arm_dataset_10M'):
    if not os.path.exists('./Data'):
        os.makedirs('./Data')
    np.savez(f'./Data/{filename}', inputs=inputs, outputs=outputs)
    print(f"Dataset saved to ./Data/{filename}.npz")

def load_dataset(filename='robot_arm_dataset.npz'):
    data = np.load(f'./Data/{filename}')
    return data['inputs'], data['outputs']


# Generate and save the dataset
inputs, outputs = generate_dataset(10_000_000)
save_dataset(inputs, outputs)