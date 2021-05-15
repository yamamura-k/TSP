import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from pointer_network import PtrNet
# 元論文： https://arxiv.org/pdf/1611.09940.pdf
# TODO
# pointer networkの実装
# glimpseの実装
# 対応して、pointer network中のdecoderをこれ用にカスタマイズする必要あり
# critic networkの実装
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
        input_dim = inputs.size(2)
        sourceL = inputs.size(1)
        
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded_inputs = []
        inpts = inputs.unsqueeze(1)

        for i in range(sourceL):
            embedded_inputs.append(torch.bmm(inpts[:, :, :, i].float(),embedding).squeeze(1))

        embedded_inputs = torch.cat(embedded_inputs).view(batch_size, sourceL, embedding.size(2))
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
    tour_len = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    
    tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    return tour_len