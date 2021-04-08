from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
"""reference
https://arxiv.org/pdf/1611.09940.pdf
https://arxiv.org/pdf/1511.06391.pdf
https://arxiv.org/pdf/1506.03134.pdf
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.LSTM = nn.LSTM(input_dim, emb_dim)
        
        self.enc = self._init_enc(emb_dim)
    
    def _init_enc(self, emb_dim):
        enc_init_hidden = Variable(torch.zeros(emb_dim), requires_grad=False)
        enc_init_c = Variable(torch.zeros(emb_dim), requires_grad=False)
        return (enc_init_hidden, enc_init_c)
    
    def forward(self, x, hidden):
        return self.LSTM(x, hidden)


class Decoder(nn.Module):
    def __init__(self, emb_dim, out_dim, max_len, ):
        super(Decoder, self).__init__()

class Attention(nn.Module):
    """base class of pointer & glimpse
    """
    def __init__(self, layer_dim, C=10, tanh_flg=False):
        super(Attention, self)__init__()
        self.proj_q = nn.Linear(layer_dim, layer_dim)# hidden vector of decoder
        self.proj_ref = nn.Linear(layer_dim, layer_dim)# hidden vector of encoder
        self.C = C
        self.tanh = nn.Tanh()
        self.tanh_flg = tanh_flg
        #self.proj_ref = nn.Conv1d(layer_dim, layer_dim, 1, 1)
        v = torch.FloatTensor(layer_dim)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1./sqrt(layer_dim)), 1./sqrt(layer_dim))
    
    def forward(self, q, ref):
        ref = ref.permute(1, 2, 0)
        q = self.proj_q(q).unsqueeze(2)
        e = self.proj_ref(ref)
        q_expand = q.repeat(1, 1, e.size(2))
        v_view = self.v.unsqueeze(0).expand(q_expand.size(0), len(self.v)).unsqueeze(1)
        u = torch.bmm(v_view, self.tanh(q_expand + e)).squeeze(1)# squeezeの意味を調べよう。
        logits = C*self.tanh(u) if self.tanh_flg else u
        return e, logits