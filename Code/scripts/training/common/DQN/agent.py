from .replay_buffer import ReplayMemory, Transition
from torch.autograd import Variable
from .model import QNetwork
import random
import torch.optim as optim
import torch
import numpy as np
from torchinfo import summary
from .curiosity_model import ICM, Inverse, Forward


BATCH_SIZE = 64


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class Agent(object):

    def __init__(self, n_states, n_actions, hidden_dim, buffer_size, curiosity, cal_r_i, device, lr, logger):
        """
        Agent class that choose action and train
        Args:
            n_states (int): input dimension
            n_actions (int): output dimension
            hidden_dim (int): hidden dimension
            buffer_size (int): number of samples
            curiosity (int): apply curiosity or not
            cal_r_i (bool): calculate intrinsic reward fpor valution or not
            device (tensor): cuda or cpu
            lr (float): learning rate
            logger: debug logger
        """

        self.device = device
        self.eta = .1
        self.curiosity = curiosity
        self.cal_r_i = cal_r_i
        # Local net
        logger.info("Initialising Local DQNetwork")
        self.q_local = QNetwork(n_states[0], n_actions,
                                hidden_dim=hidden_dim).to(self.device)
        #logger.debug(summary(self.q_local, input_size=(BATCH_SIZE, n_actions, n_states)))

        # target net
        logger.info("Initialising Target DQNetwork")
        self.q_target = QNetwork(
            n_states[0], n_actions, hidden_dim=hidden_dim).to(self.device)
        #logger.debug(summary(self.q_local, input_size=(BATCH_SIZE, n_actions, n_states)))

        self.mse_loss = torch.nn.MSELoss()
        self.optim = optim.Adam(self.q_local.parameters(), lr=lr)

        self.n_states = n_states
        self.n_actions = n_actions

        #  ReplayMemory: trajectory is saved here
        self.replay_memory = ReplayMemory(buffer_size)

        if self.curiosity != 0:
            inverse_m = Inverse(self.n_states, self.n_actions, hidden_dim)
            forward_m = Forward(self.n_states, self.n_actions,
                                inverse_m.calc_input_layer(), hidden_dim, device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            #print(inverse_m, forward_m)

        if self.cal_r_i == True:
            inverse_m = Inverse(self.n_states, self.n_actions, hidden_dim)
            forward_m = Forward(self.n_states, self.n_actions,
                                inverse_m.calc_input_layer(), hidden_dim, device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            #print(inverse_m, forward_m)

    def get_action(self, state, eps, check_eps=True):
        """
        Returns an action
        Args:
            state : 2-D tensor of shape (n, input_dim)
            eps (float): eps-greedy for exploration
        Returns: int: action index
        """
        """
        With probability EPSILON, select a random action
        Otherwise select the action with the highest Q value

        """
        sample = random.random()

        if check_eps == False or sample > eps:

            with torch.no_grad():
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:

            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device)

    def learn(self, experiences, gamma):
        """
        Prepare minibatch and train them
        Args:
        experiences (List[Transition]): batch of `Transition`
        gamma (float): Discount rate of Q_target

        """

        if len(self.replay_memory.memory) < BATCH_SIZE:
            return

        batch = Transition(*zip(*experiences))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        # calculate curiosity and add curiosity to the extrinsic rewards
        if self.curiosity != 0:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(
                state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            r_i = r_i.detach().squeeze()
            if self.curiosity == 1:
                rewards += r_i
            else:
                rewards = r_i

        # calculate curiosity for evaluation purpose without adding it to the extrinsic rewards
        avg_r_i = 0
        if self.cal_r_i == True:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(
                state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            r_i = r_i.detach().squeeze()
            # convert tensor to numpy and calculate mean
            r_i = r_i.numpy()
            avg_r_i = np.mean(r_i)

        # Compute Q(s_t, a) - the model computes Q(s_t),
        Q_expected = self.q_local(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        Q_targets_next = self.q_target(next_states).detach().max(1)[0]

        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))

        self.q_local.train(mode=True)
        # Optimize the output
        self.optim.zero_grad()

        # Compute loss
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))

        icm_loss = 0
        if self.curiosity != 0:
            icm_loss = self.ICM.update_ICM(
                loss, forward_pred_err, inverse_pred_err)

        # backpropagation of loss to NN
        loss.backward()
        self.optim.step()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.q_local, self.q_target, TAU)
        return loss.detach().cpu().numpy(), icm_loss, avg_r_i

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    # Methods to calculate the values for the potential function for PBRS
    def q_eval_value(self, states):
        states = np.asarray(states)
        states = FloatTensor([states])
        q_value = self.q_local(states).detach()
        return torch.max(q_value).numpy()

    def q_target_value(self, states):
        states = torch.from_numpy(states).float()
        q_value = self.q_target(states).detach()
        return torch.max(q_value).numpy()
