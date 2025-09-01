# This is code based on the pytorch workshop code, for my own data set (pupil change after adaptation).

# Not particularly an ideal dataset for machine learning, but doe sthe job for practise.

#%% Import necessary modules
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

#%% Changeable variables

batch_size = 16

# Bootstrapping
target_dataset_size = 1000

# Activation function
negative_slope = 0.01 # if using leaky RELU

# Optimizer
learning_rate = 0.001
momentum = 0.99 # if using SGD

# Training
n_epochs = 10 

test_size = 0.2 # 80% of sample to train, 20% to test

#%% Load CSV
df = pd.read_csv("pupilDifference.csv") # Assuming data is in the same directory as the code.

# Reset index so each row (participant) gets an ID
df_reset = df.reset_index().rename(columns={"index": "Participant"})

# Reshape into long format
df_long = df_reset.melt(
    id_vars="Participant", # keep participant ID
    var_name="Condition", # new column name for conditions
    value_name="Value" # new column name for pupil difference
)

# Encode condition labels into integers
df_long["ConditionID"] = df_long["Condition"].astype("category").cat.codes

print(df_long.head(12))
print("\nCondition mapping:")
print(dict(enumerate(df_long["Condition"].astype("category").cat.categories)))

# Features (X) = Value column
values = df_long["Value"].values.reshape(-1, 1)   # shape (N, 1)

# Labels (y) = ConditionID column
condition_class = df_long["ConditionID"].values

#%% Bootstrapping function 

# We have such a small data set, so we need to make it bigger for the sake of this practise.

def bootstrap_data(values, condition_class, target_size):
    """
    Create a larger dataset by sampling with replacement.
    
    INPUTS:
        values = np.array of shape (N, features)
        labels = np.array of shape (N,)
        target_size = int, desired number of samples in the new dataset
    
    Returns:
        boot_values = np.array of shape (target_size, features)
        boot_labels = np.array of shape (target_size,)
    """
    indices = np.random.choice(len(values), size=target_size, replace=True)
    boot_values = values[indices]
    boot_condition_class = condition_class[indices]
    return boot_values, boot_condition_class

#%% Create dataset

class PupilDataset(Dataset):
    def __init__(self, values, condition_class):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.condition_class = torch.tensor(condition_class, dtype=torch.long)  # long for classification

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx], self.condition_class[idx]

#%% Sort the datasets

boot_values, boot_condition_class = bootstrap_data(values, condition_class, target_dataset_size)

# Split bootstrapped dataset into train/test
values_train, values_test, labels_train, labels_test = train_test_split(boot_values, boot_condition_class, test_size=test_size, random_state=42)

training_dataset = PupilDataset(values_train, labels_train)
test_dataset = PupilDataset(values_test, labels_test)

# Split training set into train and validation
training_size = int(0.8 * len(training_dataset)) # 80% train
validation_size = len(training_dataset) - training_size # 20% validation
training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [training_size, validation_size])

training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(training_dataset)}")
print(f"Test samples: {len(test_dataset)}")

#%% Examine the data

# Examine individual data points
x, y = training_dataset[0] # Just the first 
print(f"First item shape: {x.shape}")
print(f"Class: {y}")

# Examine batches
batched_x, batched_y = next(iter(training_loader))
print(batched_x.shape) # 16 points
print(batched_y.shape) # 16 classes
print(batched_y) # the list of each images' class

#%% Define the model

n_features = 1  # We only have one value per data point
n_classes = len(df_long["ConditionID"].unique())  # number of conditions

class ConditionClassifier(nn.Module):
    def __init__(self):
        super(ConditionClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)   
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, n_classes) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

#%% Set up the model

# Create an instance of our model class we can train.
model = ConditionClassifier()

# Before training the network we need:
# 1: A loss function - how different the network outputs are from the intended ground truth
loss_fn = torch.nn.CrossEntropyLoss()

# 2: An optimiser - controls how the network should respond to the error from the loss function
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#%% Training loop

training_losses = []
validation_losses = []

for epoch in range(0, n_epochs): # for each epoch
    epoch_loss = 0
    model.train()
    for idx, data in enumerate(training_loader): # for each batch
        # Training ->
        inputs, labels = data
        optimizer.zero_grad() # Zero the optimiser gradients so that after each batch the error points in the correct direction toward the minimum
        outputs = model(inputs) # Forward pass over the inputs
        loss = loss_fn(outputs, labels) # Compute the loss
        loss.backward() # Compute the gradients
        optimizer.step() # Backward pass to update weights
        epoch_loss += loss.item() # Keep count of the losses from each batch for each epoch

    # Validation ->
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

    validation_loss /= len(validation_loader)
    validation_acc = correct / len(validation_dataset)
    validation_losses.append(validation_loss)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {epoch_loss/len(training_loader):.4f}, "
          f"Val Loss = {validation_loss:.4f}, "
          f"Val Acc = {validation_acc:.4f}")

    training_losses.append(epoch_loss / len(training_loader))

#%% Graph the loss curve

# Shows the performance of the model over the training epochs
plt.plot(range(1, n_epochs+1), training_losses, label="Train Loss")
plt.plot(range(1, n_epochs+1), validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

#%% Test loop

# Run the test set (not trained with) through the model

# Place the model into evaluation mode - undo with model.train() 
model.eval()

total_correct = 0
all_y = np.array([])
all_labels = np.array([])

for idx, data in enumerate(test_loader): # for each test batch
  inputs, labels = data

  outputs = model(inputs) # Forward pass

  # The greatest value in the vector represents the class
  correct_output_vector = (labels == torch.argmax(outputs, axis=1)) # Find the index, and compare to truth
  batch_correct = torch.sum(correct_output_vector)   # Sum over all correct in the batch
  batch_correct = batch_correct.numpy() # Convert to numpy (easier to work with)

  # Store all the labels and predictions for further analysis.
  all_y = np.concatenate((all_y, torch.argmax(outputs, axis=1).numpy()))
  all_labels = np.concatenate((all_labels, labels.numpy()))

  total_correct += batch_correct

accuracy = (total_correct / len(test_dataset)) * 100 
print(f"Accuracy: {np.round(accuracy, 2)}")

#%% Confusion matrix

# Inspect what we're predicting correctly / which classes the network is confusing

cm = confusion_matrix(all_labels, all_y)
ConfusionMatrixDisplay(cm).plot(xticks_rotation=90)

