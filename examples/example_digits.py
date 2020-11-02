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

y_train_np_nan = y_train_np.copy()
y_train_np_nan[:, -2:] = 0.4
y_train_np_nan

x_train = torch.FloatTensor(X_train_np)
y_train = torch.FloatTensor(y_train_np_nan)
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

def BCECLoss(output, targets, constraint_weight):
    sup_loss = supervision_loss(output, targets)
    constr_loss = constraint_loss(output)
    tot_loss = sup_loss + constraint_weight * constr_loss
    return tot_loss, sup_loss, constr_loss

def supervision_loss(output, targets):
    return torch.nn.functional.binary_cross_entropy(output, targets, reduction="sum") / len(output)

def constraint_loss(output, mu=10):

    # MAIN CLASSES
    ZERO = output[:, 0]
    ONE = output[:, 1]
    TWO = output[:, 2]
    THREE = output[:, 3]
    FOUR = output[:, 4]
    FIVE = output[:, 5]
    SIX = output[:, 6]
    SEVEN = output[:, 7]
    EIGHT = output[:, 8]
    NINE = output[:, 9]

    # ATTRIBUTE CLASSES
    ODD = output[:, 10]
    EVEN = output[:, 11]

    # here we converted each FOL rule using the product T-Norm (no-residual)
    loss_fol_product_tnorm = [
        # N(1,3,5,7,9) => ODD
        (ONE * (1. - ODD)),
        (THREE * (1. - ODD)),
        (FIVE * (1. - ODD)),
        (SEVEN * (1. - ODD)),
        (NINE * (1. - ODD)),

        # N(0,2,4,6,8) => EVEN
        (ZERO * (1. - EVEN)),
        (TWO * (1. - EVEN)),
        (FOUR * (1. - EVEN)),
        (SIX * (1. - EVEN)),
        (EIGHT * (1. - EVEN)),

        # XOR ON THE MAIN CLASSES
        mu * (
                (1 - ((ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                    SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (EIGHT) * (1 - NINE))) *
                (1 - ((1 - ZERO) * (1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (
                            1 - SEVEN) * (1 - EIGHT) * (NINE)))
        ),

        # XOR ON THE ATTRIBUTE CLASSES
        mu * (
                (EVEN) * (1 - ODD) *
                (1 - EVEN) * (ODD)
        ),
    ]

    losses = torch.sum(torch.stack(loss_fol_product_tnorm, dim=0), dim=1)
    constr_loss = torch.squeeze(torch.sum(losses, dim=0)) / len(output)
    return constr_loss


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
epoch = 4000
accuracy = 0
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_train)
    y_pred_np = y_pred.detach().numpy()

    # Compute Loss
    if accuracy < 0.9:
        tot_loss, sup_loss, constr_loss = BCECLoss(y_pred, y_train, constraint_weight=0.01)
    else:
        tot_loss, sup_loss, constr_loss = BCECLoss(y_pred, y_train, constraint_weight=0.5)

    # compute accuracy
    y_pred_d = (y_pred > 0.5).detach().numpy()
    accuracy = ((y_pred_d[:, :-2] == y_train_np[:, :-2]).sum(axis=1) == y_train_np[:, :-2].shape[1]).mean()
    accuracy2 = ((y_pred_d == y_train_np).sum(axis=1) == y_train_np.shape[1]).mean()

    print(f'Epoch {epoch + 1}: '
          f'total loss: {tot_loss.item():.4f} '
          f'| supervision loss: {sup_loss.item():.4f} '
          f'| constraint loss: {constr_loss.item():.4f} '
          f'| accuracy: {accuracy:.4f} '
          f'| accuracy2: {accuracy2:.4f}')

    # Backward pass
    tot_loss.backward()
    optimizer.step()

# %%

y_pred = model(x_test)

# compute accuracy
y_pred_round = y_pred > 0.5
accuracy = ((y_pred_round == y_test).sum(dim=1) == y_test.shape[1]).to(torch.float).mean()

print(f'accuracy: {accuracy.detach():.4f}')

y_pred = model(x_train)
(y_pred[:, -2] > 0.5) == (y_pred[:, -1] > 0.5)
print((y_pred[-3] > 0.5).to(torch.float).detach().numpy())
print(y_train_np[-3])

pred = (y_pred[-3] > 0.5).to(torch.float).detach().numpy()
y_pred[-3][2] * (1 - y_pred[-3][-1])
1 - y_pred[-3][-1]

## Verification

# %%

y_pred_zero = y_pred_round[y_pred_round[:, 0] == 1]
n_violations = (y_pred_zero[:, 0] != y_pred_zero[:, -1]).sum() / len(y_pred_zero)
print(f'"0 implies EVEN": {n_violations * 100:.2f}% violations')

# %%

n_violations = (y_pred_round[:, -1] == y_pred_round[:, -2]).sum() / len(y_pred_round)
print(f'"XOR(EVEN, ODD)": {n_violations * 100:.2f}% violations')

# %%

n_violations = (y_pred_round[:, :-2].sum(axis=1) > 1).sum() / len(y_pred_round)
print(f'"XOR(0,1,2,3,4,5,6,7,8,9)": {n_violations * 100:.2f}% violations')

