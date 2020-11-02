from itertools import product

import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)

print(f'X shape: {X.shape}\nClasses: {np.unique(y)}')

# show the first ten images
# figs = X[:10].reshape((10, 8, 8))
# plt.figure(figsize=[7, 6])
# for i, fig in enumerate(figs):
#     plt.subplot(5, 5, i + 1)
#     sns.heatmap(fig, cbar=False)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

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


class ExplainNet(torch.nn.Module):
    def __init__(self, D_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ExplainNet, self).__init__()
        self.linear = torch.nn.Linear(D_in, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear(x)
        y_pred = torch.sigmoid(h)
        return y_pred


def BCECLoss(output, targets, constraint_weight):
    sup_loss = supervision_loss(output, targets)
    # constr_loss = constraint_loss(output)
    # tot_loss = sup_loss + constraint_weight * constr_loss
    return sup_loss #, sup_loss, constr_loss

def supervision_loss(output, targets):
    return torch.nn.functional.binary_cross_entropy(output, targets, reduction="sum") / len(output)


din, dh, dout = x_train.shape[1], 20, y_train.shape[1]
model = FeedForwardNet(din, dh, dout)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
epoch = 500
accuracy = 0
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    tot_loss = BCECLoss(y_pred, y_train, constraint_weight=0.01)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy()
    accuracy = ((y_pred_d == y_train_np).sum(axis=1) == y_train_np.shape[1]).mean()

    print(f'Epoch {epoch + 1}: '
          f'total loss: {tot_loss.item():.4f} '
          f'| accuracy: {accuracy:.4f} ')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

# %%

y_pred = model(x_test)

# compute accuracy
y_pred_round = y_pred > 0.5
accuracy = ((y_pred_round == y_test).sum(dim=1) == y_test.shape[1]).to(torch.float).mean()

print(f'accuracy: {accuracy.detach():.4f}')


y_pred_train = (model(x_train).detach().numpy() > 0.5).astype(float)
y_pred_test = (model(x_test).detach().numpy() > 0.5).astype(float)

x_concepts_train_np, y_concepts_train_np = y_pred_train[:, :-1], y_pred_train[:, -1]
x_concepts_test_np, y_concepts_test_np = y_pred_test[:, :-1], y_pred_test[:, -1]

x_concepts_train, y_concepts_train = torch.FloatTensor(x_concepts_train_np), torch.FloatTensor(y_concepts_train_np)
x_concepts_test, y_concepts_test = torch.FloatTensor(x_concepts_test_np), torch.FloatTensor(y_concepts_test_np)

D_in = y_pred_train.shape[1] - 1
enet = ExplainNet(D_in)
print(enet)

optimizer = torch.optim.Adam(enet.parameters(), lr=0.01)
enet.train()
epoch = 500
accuracy = 0
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = enet(x_concepts_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    tot_loss = BCECLoss(y_pred, y_concepts_train, constraint_weight=0.01)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy().ravel()
    accuracy = (y_pred_d == y_concepts_train_np).mean()

    print(f'Epoch {epoch + 1}: '
          f'total loss: {tot_loss.item():.4f} '
          f'| accuracy: {accuracy:.4f} ')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

weight_mask = enet.linear.weight > 0.5
mask_size = torch.sum(weight_mask)
truth_table_np = np.zeros((2**mask_size, D_in))
option_list = np.array(list(product((True, False), repeat=mask_size)))
j = 0
for i, is_valid in enumerate(weight_mask.detach().numpy().ravel()):
    if is_valid:
        truth_table_np[:, i] = option_list[:, j]
        j += 1
truth_table = torch.FloatTensor(truth_table_np)

enet.eval()
constraints_truth = enet(truth_table)

# truth_table_list = []
# for i in range(truth_table.shape[0]):
#     truth_table_list.append(list(option_list[i, :].astype(int)))


# from sympy.logic import SOPform
# from sympy import symbols
#
# zero, two, four, six, eight = symbols('zero two four six eight')
# symb = [zero, two, four, six, eight]
# SOPform(symb, truth_table_list)