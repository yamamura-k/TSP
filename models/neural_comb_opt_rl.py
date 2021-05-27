import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import numpy as np

from models.pointer_network import Encoder
# 元論文： https://arxiv.org/pdf/1611.09940.pdf
# TODO
# pointer networkの実装
# glimpseの実装
# 対応して、pointer network中のdecoderをこれ用にカスタマイズする必要あり
# critic networkの実装
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_glimpse=0):
        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        # self.context_lin = nn.Linear(input_dim, hidden_dim)# 論文はこっちだけど、これにすると途中で行列サイズが合わなくなる
        self.context_lin = nn.Conv1d(input_dim, hidden_dim, 1, 1)# number of input channel, number of output channel, kernel size, stlide
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        nn.init.uniform_(self.V, -1, 1)
    
    def forward(self, input_hidden_state, context, selection_mask):
        # batchsize, hidden_dim, length_of_sequence
        _input = self.input_lin(input_hidden_state).unsqueeze(2).expand(-1, -1, context.size(1))

        context = context.permute(0, 2, 1)
        _context = self.context_lin(context)

        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)# 逆伝播計算でVがnanになっているように見える

        # batchsize, length_of_sequence
        attention = torch.bmm(V, self.tanh(_input + _context)).squeeze(1)

        if len(attention[selection_mask]) > 0:
            attention[selection_mask] = self.inf[selection_mask]
        
        alpha = self.softmax(attention)

        hidden_state = torch.bmm(_context, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha
    
    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, max_decode_len, use_cuda=False):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.max_decode_len = max_decode_len
        self.use_cuda = use_cuda

        self.input2hidden = nn.Linear(embedding_dim, 4*hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, 4*hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim*2, hidden_dim)
        self.attention = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, decoder_input, hidden, context):
        batch_size = embedded_inputs.size(0)
        input_length = self.max_decode_len

        selection_mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.attention.init_inf(selection_mask.size())

        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = torch.ones(batch_size)
        if self.use_cuda:
            outputs = outputs.cuda()# gpuを使う場合
        pointers = []

        def step(x, hidden):
            # 通常のLSTM
            h, c = hidden
            gates = self.input2hidden(x) + self.hidden2hidden(h)

            _input, forget, cell, out = gates.chunk(4, 1)

            _input = torch.sigmoid(_input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            context_state = forget*c + _input*cell
            hidden_state = out*torch.tanh(context_state)

            hidden_t, output = self.attention(hidden_state, context, torch.eq(selection_mask, 0))
            hidden_t = torch.tanh(self.hidden2out(torch.cat((hidden_t, hidden_state), 1)))

            return hidden_t, context_state, output

        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            masked_outputs = outs*selection_mask

            # 確率最大のものを選択する
            max_probs, indices = masked_outputs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            selection_mask = selection_mask*(1 - one_hot_pointers)

            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs = outputs*max_probs
            pointers.append(indices.unsqueeze(1))
        
        #outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden

class PtrNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_lstm_layers, dropout, max_decode_len, bidirectional=False, use_cuda=False):
        super(PtrNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_decode_len = max_decode_len
        self.bidir = bidirectional
        
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim, num_lstm_layers, dropout, bidirectional)

        self.decoder = Decoder(embedding_dim, hidden_dim, max_decode_len, use_cuda)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_length = self.max_decode_len

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size*input_length, -1)
        
        # [batch_size x input_length x embedding_dim]
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        # [batch_size x input_length x hidden_dim]
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, encoder_hidden0)

        if self.bidir:
            decoder_hidden0 = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1), torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])

        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs, decoder_input0, decoder_hidden0, encoder_outputs)

        return outputs, pointers
    
class Critic(nn.Module):
    """
    - LSTM encoder (1個, poniter networkと同様)
    - LSTM process block (1個, Vinyals et al., 2015a と同様)
    - 2-layer ReLU decoder
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers=5, dropout=0.5, n_process_block_iters=4, bidirectional=False):
        super(Critic, self).__init__()
        self.n_process_block_iters = n_process_block_iters
        self.encoder = Encoder(embedding_dim, hidden_dim, n_layers, dropout, bidirectional)
        self.process_block = Attention(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.softmax = nn.Softmax()
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        encoder_hidden0 = self.encoder.init_hidden(inputs)
        # [batch_size x input_length x hidden_dim]
        encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden0)
        
        if self.bidir:
            process_block_state = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1), torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            process_block_state = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        
        for _ in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, encoder_outputs)
            process_block_state = torch.bmm(ref, self.softmax(logits).unsqueeze(2)).squeeze(2)

        (outputs, pointers), decoder_hidden = self.decoder(inputs, decoder_input0, process_block_state, encoder_outputs)
        
        return (outputs, pointers), decoder_hidden



class NeuralCombOptRL(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, max_decode_len, num_lstm_layers, dropout, objective, bidirectional=False, use_cuda=False, is_train=False):
        super(NeuralCombOptRL, self).__init__()
        self.objective = objective
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        self.is_train = is_train
        self.max_decode_len = max_decode_len
        self.actor_net = PtrNet(embedding_dim, hidden_dim, num_lstm_layers, dropout, max_decode_len, bidirectional=bidirectional, use_cuda=use_cuda)

    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_dim = inputs.size(2)
        sourceL = inputs.size(1)
        probs_, action_idxs = self.actor_net(inputs)
        actions = []
        #breakpoint()
        for i, action_id in enumerate(action_idxs):
            actions.append(inputs[i, action_id.data, :])
        
        if False:# self.is_train:
            probs = []
            for i, (prob, action_id) in enumerate(zip(probs_, action_idxs)):
                probs.append(probs_[i, action_id.data])
        
        else:
            probs = probs_
        R = self.objective(actions, self.use_cuda)

        return R, probs, actions, action_idxs

def reward_tsp(sample_solution, USE_CUDA=False):
    batch_size = len(sample_solution)
    n = sample_solution[0].size(0)
    tour_len = Tensor(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()
    for i in range(batch_size):
        for j in range(n-1):
            tour_len[i] += torch.norm(sample_solution[i][j] - sample_solution[i][j+1])
        tour_len[i] += torch.norm(sample_solution[i][n-1] - sample_solution[i][0])
    return tour_len
