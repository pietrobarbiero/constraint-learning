import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product
import pandas as pd
import torch.nn.utils.prune as prune


X, y = load_digits(return_X_y=True)
print(f'X shape: {X.shape}\nClasses: {np.unique(y)}')

enc = OneHotEncoder()
y1h = enc.fit_transform(y.reshape(-1, 1)).toarray()
print(f'Before: {y.shape}\nAfter: {y1h.shape}')

y2 = np.zeros((len(y), 2))
for i, yi in enumerate(y):
    if yi % 2:
        y2[i, 0] = 1
    else:
        y2[i, 1] = 1
y1h2 = np.hstack((y1h, y2))

print(f'Target vector shape: {y1h2.shape}')
for i in range(10):
    print(f'Example ({y[i]}): {y1h2[i]}')

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y1h2, test_size=0.33, random_state=42)

x_train = torch.FloatTensor(X_train_np)
y_train = torch.FloatTensor(y_train_np)
x_test = torch.FloatTensor(X_test_np)
y_test = torch.FloatTensor(y_test_np)


class FeedForwardNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FeedForwardNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        h = torch.nn.functional.relu(h)
        h = self.linear2(h)
        h = torch.nn.functional.relu(h)
        h = self.linear3(h)
        y_pred = torch.sigmoid(h)
        return y_pred


din, dh, dout = x_train.shape[1], 20, y_train.shape[1]
model = FeedForwardNet(din, dh, dout)

print(model)
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
epoch = 2000
for epoch in range(epoch):
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    tot_loss = loss(y_pred, y_train)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy()
    accuracy = ((y_pred_d == y_train_np).sum(axis=1) == y_train_np.shape[1]).mean()

    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}: '
              f'total loss: {tot_loss.item():.4f} '
              f'| accuracy: {accuracy:.4f} ')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

y_pred = model(x_test)

# compute accuracy
y_pred_round = (y_pred > 0.5).to(torch.float).detach().numpy()
accuracy = ((y_pred_round == y_test_np).sum(axis=1) == y_test_np.shape[1]).mean()

print(f'accuracy: {accuracy:.4f}')


class ExplainEven(torch.nn.Module):
    def __init__(self, D_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ExplainEven, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 10)
        self.linear2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        h = self.linear2(h)
        y_pred = torch.sigmoid(h)
        return y_pred


y_pred_train = model(x_train).detach().numpy().astype(float)
y_pred_test = model(x_test).detach().numpy().astype(float)

x_concepts_train_np, y_concepts_train_np = y_pred_train[:, :-1], y_pred_train[:, -1]
x_concepts_test_np, y_concepts_test_np = y_pred_test[:, :-1], y_pred_test[:, -1]

x_concepts_train, y_concepts_train = torch.FloatTensor(x_concepts_train_np), torch.FloatTensor(y_concepts_train_np)
x_concepts_test, y_concepts_test = torch.FloatTensor(x_concepts_test_np), torch.FloatTensor(y_concepts_test_np)

D_in = y_pred_train.shape[1] - 1
even_net = ExplainEven(D_in)
print(even_net)

optimizer = torch.optim.Adam(even_net.parameters(), lr=0.01)
even_net.train()
epoch = 500
accuracy = 0
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = even_net(x_concepts_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    tot_loss = loss(y_pred.squeeze(), y_concepts_train) + \
               0.08 * \
               even_net.linear1.weight.norm(1) * \
               even_net.linear2.weight.norm(1)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy().ravel()
    accuracy = (y_pred_d == (y_concepts_train_np > 0.5)).mean()

    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}: '
              f'total loss: {tot_loss.item():.4f} '
              f'| accuracy: {accuracy:.4f} ')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

# Pruning
for i, (module) in enumerate(even_net._modules.items()):
    mask = torch.ones(module[1].weight.shape)
    param_absneg = -torch.abs(module[1].weight)
    idx = torch.topk(param_absneg, k=param_absneg.shape[1] - 2)[1]
    for i in range(len(idx)):
        mask[i, idx[i]] = 0
    prune.custom_from_mask(module[1], name="weight", mask=mask)

# Tuning
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = even_net(x_concepts_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    tot_loss = loss(y_pred.squeeze(), y_concepts_train)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy().ravel()
    accuracy = (y_pred_d == (y_concepts_train_np > 0.5)).mean()

    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}: '
              f'total loss: {tot_loss.item():.4f} '
              f'| accuracy: {accuracy:.4f} ')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

weights, bias = [], []
for i, (module) in enumerate(even_net._modules.items()):
    weights.append(module[1].weight.detach().numpy())
    bias.append(module[1].bias.detach().numpy())


from intoCNF_with_prints import booleanConstraint

f = booleanConstraint(weights, bias)
print(f)

from sympy.logic import simplify_logic

sf = simplify_logic(f[0])
print(sf)

a = 1
