# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Pavithra K

### Register Number:212224240112

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

data = {
    "Input": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "Output": [5,9,14,18,23,27,32,36,41,45,50,54,59,63,68]
}

df = pd.DataFrame(data)

print(df)

x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=52)


scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)


xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)


class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.network(x)
        
model=neuralNetwork()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=5000
losses=[]


for i in range(epochs):
    optimizer.zero_grad()
    predictions=model(xt)
    loss=criterion(predictions,yt)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"epoch: {i}, loss: {loss.item()}")
        losses.append(loss.item())
        
        
new_data=torch.FloatTensor([[16]])
new_data=torch.Tensor(scale1.transform(new_data))
predictions=model(new_data)
print(predictions.item())


xst = torch.FloatTensor(xst)
yst = torch.FloatTensor(yst)
with torch.no_grad():
    predictions_l=model(xst)
    test_loss=criterion(predictions_l,yst)
    print(f"test loss: {test_loss.item()}")

    
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.show()


```

### Dataset Information
<img width="282" height="322" alt="image" src="https://github.com/user-attachments/assets/775cd45a-5e92-40e0-b689-f70070b0887e" />

### OUTPUT
<img width="763" height="540" alt="image" src="https://github.com/user-attachments/assets/efb62b4c-8f40-4976-9827-37495427c0be" />

### Training Loss Vs Iteration Plot
<img width="692" height="502" alt="image" src="https://github.com/user-attachments/assets/be8e3482-1511-4d05-bcbf-e7beb6487145" />

### New Sample Data Prediction

<img width="406" height="45" alt="image" src="https://github.com/user-attachments/assets/204df064-c176-4286-860c-01e7aa25c92b" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
