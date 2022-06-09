import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")


class Inverse(nn.Module):
    """
    first submodel: encodes the state and next state into feature space.
    second submodel: the inverse approximates the action taken by the given state and next state in feature size
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Inverse, self).__init__()
        self.state_size = input_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim[0], hidden_dim//2),
                                     nn.ELU())

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def calc_input_layer(self):
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return x.flatten().shape[0]

    def forward(self, enc_state, enc_state1):
        """
        Input: state s and state s' as torch Tensors with shape: (batch_size, state_size)
        Output: action probs with shape (batch_size, action_size)
        """
        x = torch.cat((enc_state, enc_state1), dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x


class Forward(nn.Module):

    def __init__(self, input_dim, output_dim, output_size, hidden_dim, device=device):
        super(Forward, self).__init__()
        self.action_size = output_dim
        self.device = device
        self.forwardM = nn.Sequential(nn.Linear(output_size+self.action_size, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, output_size))

    def forward(self, state, action):
        # One-hot-encoding for the actions
        ohe_action = torch.zeros(
            action.shape[0], self.action_size).to(self.device)
        indices = torch.stack((torch.arange(action.shape[0]).to(
            self.device), action.squeeze().long()), dim=0)
        indices = indices.tolist()
        ohe_action[indices] = 1.

        # concat state embedding and encoded action
        x = torch.cat((state, ohe_action), dim=1)
        return self.forwardM(x)


class ICM(nn.Module):
    def __init__(self, inverse_model, forward_model, learning_rate=1e-3, lambda_=0.1, beta=0.2, device=device):
        super(ICM, self).__init__()
        self.inverse_model = inverse_model.to(device)
        self.forward_model = forward_model.to(device)

        self.forward_scale = 1.
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = beta
        self.lambda_ = lambda_
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(list(self.forward_model.parameters(
        ))+list(self.inverse_model.parameters()), lr=1e-3)

    def calc_errors(self, state1, state2, action):
        """
        Input: Torch Tensors state s, state s', action a with shapes
        s: (batch_size, input_dim)
        s': (batch_size, input_dim)
        a: (batch_size, 1)

        """
        enc_state1 = self.inverse_model.encoder(
            state1).view(state1.shape[0], -1)
        enc_state2 = self.inverse_model.encoder(
            state2).view(state2.shape[0], -1)

        # forward error
        forward_pred = self.forward_model(enc_state1.detach(), action)
        forward_pred_err = 1/2 * self.forward_loss(forward_pred, enc_state2.detach()).sum(
            dim=1).unsqueeze(dim=1)

        # prediction error
        pred_action = self.inverse_model(enc_state1, enc_state2)
        inverse_pred_err = self.inverse_loss(
            pred_action, action.flatten().long()).unsqueeze(dim=1)

        return forward_pred_err, inverse_pred_err

    def update_ICM(self, q_loss, forward_err, inverse_err):
        self.optimizer.zero_grad()

        # ICM Loss
        loss = ((1. - self.beta)*inverse_err + self.beta*forward_err).mean()

        # overall Loss
        loss_ = ((1. - self.beta)*inverse_err + self.beta*forward_err)
        loss_ = loss_.sum() / loss_.flatten().shape[0]
        loss_ = loss_ + self.lambda_ * q_loss

        loss.backward(retain_graph=True)
        clip_grad_norm_(self.inverse_model.parameters(), 1)
        clip_grad_norm_(self.forward_model.parameters(), 1)
        self.optimizer.step()
        return loss.detach().cpu().numpy()
