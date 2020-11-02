import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.special import softmax
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import r2_score
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


def BCECLoss(output, targets, constraint_weight):
    c_sup = output.shape[0]
    sup_loss = torch.nn.functional.binary_cross_entropy(output, targets, reduction="sum")
    norm_sup_loss = sup_loss / c_sup
    constr_loss = constraint_loss(output) / c_sup
    tot_loss = norm_sup_loss + constraint_weight * constr_loss
    return tot_loss, norm_sup_loss, constr_loss


def constraint_loss(output, mu=10, sum=True):
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
        # 0) N(1,3,5,7,9) => ODD
        (ONE * (1. - ODD)),
        (THREE * (1. - ODD)),
        (FIVE * (1. - ODD)),
        (SEVEN * (1. - ODD)),
        (NINE * (1. - ODD)),

        # 1) N(0,2,4,6,8) => EVEN
        (ZERO * (1. - EVEN)),
        (TWO * (1. - EVEN)),
        (FOUR * (1. - EVEN)),
        (SIX * (1. - EVEN)),
        (EIGHT * (1. - EVEN)),

        # 16) XOR ON THE MAIN CLASSES
        mu * (
                (1 - ((ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (FOUR) * (1 - FIVE) * (1 - SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (FIVE) * (1 - SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (SIX) * (1 - SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (SEVEN) * (1 - EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (SEVEN) * (EIGHT) * (1 - NINE))) *
                (1 - ((1 - ONE) * (1 - TWO) * (1 - THREE) * (1 - FOUR) * (1 - FIVE) * (1 - SIX) * (SEVEN) * (1 - EIGHT) * (NINE)))
              ),

        # 17) XOR ON THE ATTRIBUTE CLASSES
        mu * (
                (EVEN) * (1 - ODD) *
                (1 - EVEN) * (ODD)
        ),
    ]

    if sum:
        losses = torch.sum(torch.stack(loss_fol_product_tnorm, dim=0), dim=1)
    else:
        losses = torch.stack(loss_fol_product_tnorm, dim=0)

    loss_sum = torch.squeeze(torch.sum(losses, dim=0))
    return loss_sum


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

    # criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.eval()
    y_pred = model(x_train)
    tot_loss, norm_sup_loss, constr_loss = BCECLoss(y_pred, y_train, 1)
    print('Test loss before training', tot_loss.item())

    model.train()
    epoch = 1000
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        tot_loss, norm_sup_loss, constr_loss = BCECLoss(y_pred, y_train, 0.2)

        # compute accuracy
        y_pred_d = y_pred > 0.5
        accuracy = ((y_pred_d == y_train).sum(dim=1) == y_train.shape[1]).to(torch.float).mean()

        print(f'Epoch {epoch}: '
              f't-loss: {tot_loss.item()} '
              f'| s-loss: {norm_sup_loss.item()} '
              f'| c-loss: {constr_loss.item()} '
              f'| train accuracy: {accuracy.detach()}')

        # Backward pass
        tot_loss.backward()
        optimizer.step()


    # y_pred_np = y_pred.detach().numpy()
    # y_pred_round = np.round(y_pred_np)
    #
    # scores = np.corrcoef(y_pred_round.T)
    #
    # scores_even = scores[-1, :-1]
    # # scores_even = scores[0, 1:]
    # alphas = softmax(scores_even)
    # betas = softmax(1-scores_even)
    # attention = np.stack((alphas, betas))
    #
    # plt.figure()
    # sns.heatmap(attention)
    # plt.show()

    return


if __name__ == "__main__":
    main()
