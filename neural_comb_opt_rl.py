import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from pointer_network import PtrNet, Encoder
# 元論文： https://arxiv.org/pdf/1611.09940.pdf
# TODO
# pointer networkの実装
# glimpseの実装
# 対応して、pointer network中のdecoderをこれ用にカスタマイズする必要あり
# critic networkの実装
class Decoder(nn.Module):
    def __init__(self):
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        return self.relu2(self.relu1(inputs))
    
class Critic(nn.Module):
    """
    - LSTM encoder (1個, poniter networkと同様)
    - LSTM process block (1個, Vinyals et al., 2015a と同様)
    - 2-layer ReLU decoder
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers=5, dropout=0.5, bidirectional=False):
        super(Critic, self).__init__()
        self.encoder = Encoder(embedding_dim, hidden_dim, n_layers, dropout, bidirectional)
        self.decoder = Decoder()
    
    #def forward(self, )


class NeuralCombOptRL(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_lstm_layers, dropout, objective, bidirectional=False, use_cuda=False, is_train=False):
        super(NeuralCombOptRL, self).__init__()
        self.objective = objective
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        self.is_train = is_train
        self.actor_net = PtrNet(embedding_dim, hidden_dim, num_lstm_layers, dropout, bidirectional=bidirectional)

        embedding = torch.FloatTensor(input_dim, embedding_dim)
        if use_cuda:
            embedding = embedding.cuda()
        self.embedding = Parameter(embedding)
        self.embedding.data.uniform_(-(1./pow(embedding_dim, 0.5)), 1./pow(embedding_dim, 0.5))
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = inputs.size(2)
        
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded_inputs = []
        inpts = inputs.unsqueeze(1)

        for i in range(sourceL):
            embedded_inputs.append(torch.bmm(inpts[:, :, :, i].float(),embedding).squeeze(1))

        embedded_inputs = torch.cat(embedded_inputs).view(batch_size, embedding.size(2), sourceL)
        probs_, action_idxs = self.actor_net(embedded_inputs)

        actions = []
        inputs_ = inputs.transpose(1, 2)

        for action_id in action_idxs:
            actions.append(inputs_[list(range(batch_size)), action_id.data, :])
        
        if self.is_train:
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[list(range(batch_size)), action_id.data])
        
        else:
            probs = probs_
        
        R = self.objective(actions, self.use_cuda)

        return R, probs, actions, action_idxs

def reward_tsp(sample_solution, USE_CUDA=False):
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Tensor(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    
    tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    return tour_len