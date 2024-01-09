import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load CSV data
data = pd.read_csv('your_data.csv')

# Data preprocessing
# Assuming columns 'azimuth' and 'altitude' are present
# ... perform necessary preprocessing steps here

# Split data into features and target
X = data[['azimuth', 'altitude']].values
y = data[['azimuth', 'altitude']].values  # Assuming predicting same features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create a custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Create DataLoader for efficient batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Build the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Using only the last time step output
        return out

input_size = 2  # Number of features (azimuth, altitude)
hidden_size = 64
output_size = 2  # Predicting azimuth and altitude

model = RNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    model.eval()
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs.unsqueeze(1))
        test_loss += criterion(outputs, labels).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")

# Make predictions
predictions = []
with torch.no_grad():
    model.eval()
    for inputs, _ in test_loader:
        outputs = model(inputs.unsqueeze(1))
        predictions.append(outputs.numpy())

predictions = np.concatenate(predictions)

# Print the first few rows of predictions
predictions_df = pd.DataFrame({'predicted_azimuth': predictions[:, 0], 'predicted_altitude': predictions[:, 1]})
print(predictions_df.head())
