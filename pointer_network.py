import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
"""
メモ
unsqueeze : 次元を追加
expand : 指定した大きさにtensorを変形する。（要素の値は元々入ってたものが使われる）
repeat : 元々入っていたデータが、指定された回数コピーされる
cat : 複数のtensorをくっつける
chunk : tensorを指定したchunkの数に分割する

参考：
Sequence to Sequence Learning with Neural Networks(https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
Pointer Networks(https://arxiv.org/pdf/1506.03134.pdf)

"""
class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, bidirectional):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidirectional else hidden_dim
        self.n_layers = n_layers*2 if bidirectional else n_layers
        
        self.bidir = bidirectional
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, embedded_inputs, hidden):
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        outputs, hidden = self.lstm(embedded_inputs, hidden)
        return outputs.permute(1, 0, 2), hidden
    
    def init_hidden(self, embedded_inputs):
        batch_size = embedded_inputs.size(0)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)

        return h0, c0

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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

        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

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
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim

        self.input2hidden = nn.Linear(embedding_dim, 4*hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, 4*hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim*2, hidden_dim)
        self.attention = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, embedded_inputs, decoder_input, hidden, context):
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        selection_mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.attention.init_inf(selection_mask.size())

        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
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

            max_probs, indices = masked_outputs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            selection_mask = selection_mask*(1 - one_hot_pointers)

            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
        
        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden

class PtrNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_lstm_layers, dropout, bidirectional=False):
        super(PtrNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidirectional
        
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim, num_lstm_layers, dropout, bidirectional)

        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size*input_length, -1)
        
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, encoder_hidden0)

        if self.bidir:
            decoder_hidden0 = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1), torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])

        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs, decoder_input0, decoder_hidden0, encoder_outputs)

        return outputs, pointers
