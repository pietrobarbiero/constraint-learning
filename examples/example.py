import torch
from scipy.special import softmax
from sklearn.datasets import load_digits
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
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
        # y_pred = torch.round(y_pred)
        return y_pred


class ManualLinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x


def main():

    X, y = load_digits(return_X_y=True)
    # one-hot encoding of y
    enc = OneHotEncoder()
    y1h = enc.fit_transform(y.reshape(-1, 1)).toarray()
    # create even-odd labels
    y2 = np.zeros((len(y), 2))
    for i, yi in enumerate(y):
        if yi % 2:
            y2[i, 0] = 1
        else:
            y2[i, 1] = 1
    y1h2 = np.hstack((y1h, y2))
    # y1h2 = y1h
    x_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y1h2)

    din, dh, dout = X.shape[1], 20, y1h2.shape[1]
    model = TwoLayerNet(din, dh, dout)

    # def my_loss(output, target):
    #     loss = torch.abs(output-target)
    #     loss = torch.sum(loss, dim=1)
    #     loss = torch.mean(loss)
    #     return loss
    # criterion = my_loss

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.eval()
    y_pred = model(x_train)
    before_train = criterion(y_pred, y_train)
    print('Test loss before training', before_train.item())

    model.train()
    epoch = 1000
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred, y_train)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))  # Backward pass
        loss.backward()
        optimizer.step()

    y_pred_np = y_pred.detach().numpy()
    y_pred_round = np.round(y_pred_np)

    scores = np.corrcoef(y_pred_round.T)

    # scores_even = scores[-1, :-1]
    scores_even = scores[0, 1:]
    alphas = softmax(scores_even)
    betas = softmax(1-scores_even)
    attention = np.stack((alphas, betas))

    plt.figure()
    sns.heatmap(attention)
    plt.show()

    return


if __name__ == "__main__":
    main()
