# This is code from a Pytorch workshop for machine learning.

# It uses the Fashion MNIST dataset

#%% Import necessary modules
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% Changeable variables

batch_size = 32

# Activation function
negative_slope = 0.01 # if using leaky RELU

# Optimizer
learning_rate = 0.001
momentum = 0.99 # if using SGD

# Training
n_epochs = 5 

#%% Transform and load data

# The dataset has 28x28 numpy array per datapoint (an image of a piece of clothing)
# But, pytorch needs a particular form called Pytorch Tensor
# So, transform the data points as we load them in

# Make the transform
transform = transforms.Compose([transforms.ToTensor()])
transforms.Normalize((0.5,), (0.5,))

# Load in the data sets (different sets for the training and testing stages!!)
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
class_labels = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Split training set into train and validation
train_size = int(0.8 * len(training_set)) # 80% train
validation_size = len(training_set) - train_size # 20% validation
train_dataset, validation_dataset = torch.utils.data.random_split(training_set, [train_size, validation_size])

#%% Examine individual data points

x, y = training_set[0] # Just the first image
print(f"First item shape: {x.shape}")
print(f"Class: name - {class_labels[y]} / number - {y}")

# The data points are images, so the tensor has a shape of (C, H, W)
# C = number of channels (1 = greyscale)
# H = height
# W = width

# Turn the Pytorch tensor into a numpy array, as matplotlib can't handle Tensors.
img = x.numpy()
img = np.squeeze(img)

height = np.size(img, 0)
width = np.size(img, 1)

plt.imshow(img, cmap="Greys")
plt.show()

#%% Batch data

# We want to be able to handle lots of data at once
# So, we can make 'batches' of data

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


batched_x, batched_y = next(iter(training_loader))
print(batched_x.shape) # 32 images
print(batched_y.shape) # 32 classes
print(batched_y) # the list of each images' class

# The data has a shape of (N, C, H, W)
# N = batch size

# %% Define the model functions

# Define the architecture and inference function
# Here we are using a simple dense neural network

# PyTorch models inherit from torch.nn.Module
class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        """
        Define the model in the init function of a class which inherits from nn.Module.

        The number of inputs in first layer needs to match the dimensionality of the data. 
        We're mapping each pixel of the input image into a single neuron. 
        As the image has a 28 by 28 pixel resolution, we need 28 * 28 = 784 input neurons.
            
        """
        super(FashionMNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(height * width, 256)   
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10) 

# Define our trainable layers as instance variables of the class
# We are only using 3 layers: input, hidden and output

# In the forward function, these layers are called in sequence
# Output of previous layer becomes input to the next

# Wrap the output of each layer in an activation function
# This adds non-linearity to the neural network so can approximate non-linear functions
# Using ReLU, all outputs below 0 are set to 0

    def forward(self, x):
        """
        Forward propagation function. 
        Takes an input and performs the layer operations defined.
        The activation function is a Rectified Linear Unit.

        INPUTS:
            x = The input batch

        Returns: The output after passing through the layers specified
        
        """
        # Reshape the 2D image data into a 1D vector
        x = x.view(-1, height * width)

        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        
        #x = F.leaky_relu(self.fc1(x), negative_slope = negative_slope)
        #x = F.leaky_relu(self.fc2(x), negative_slope = negative_slope)
        #x = self.fc3(x)
        
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))   
        x = self.fc5(x)
        
        # Not much difference between activators for this dataset, gelu maybe slightly better
        
        # Not much difference in number of layers (probably as simple dataset) between 1 hidden layer and 3 hidden layers
        
        return x

#%% Set up the model

# Create an instance of our model class we can train.
model = FashionMNISTClassifier()

# Before training the network we need:
    
# 1: A loss function - how different the network outputs are from the intended ground truth
# Here we are using cross entropy - standard for multi class classification
loss_fn = torch.nn.CrossEntropyLoss()

# 2: An optimiser - controls how the network should respond to the error from the loss function
# Here we are using Stochastic Gradient Descent
# We are optimising the parameters of the model
# Change the learning rate = how much the weights are changed in response to the error
# Change the momentum = how much to keep going in one direction

#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# SGD converges at 3rd/4th epoch
# Adam converges at 2nd

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

# The output for a given input is a 10-dimensional tensor
# Each index in the tensor represents a class
# The value at that index represents how strongly the network believes the input is that class

# To get at an accuracy score, we can take the argmax of each 10D tensor and compare this to the labels

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

accuracy = (total_correct / len(test_set)) * 100 
print(f"Accuracy: {np.round(accuracy, 2)}")

#%% Confusion matrix

# Inspect what we're predicting correctly / which classes the network is confusing

cm = confusion_matrix(all_labels, all_y)
ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(xticks_rotation=90)
