from math import sqrt
import torch
import torch.nn as nn
import torch.functional as F
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
    def __init__(self, emb_dim, out_dim, max_len, C, tanh_flg):
        super(Decoder, self).__init__()
            terminating_symbol,
            decode_type,
            n_glimpses=1,
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol 
        self.decode_type = decode_type

        self.input_weights = nn.Linear(emb_dim, 4 * out_dim)
        self.out_weights = nn.Linear(out_dim, 4 * out_dim)

        self.pointer = Attention(out_dim, tanh_flg=tanh_flg, C=C)
        self.glimpse = Attention(out_dim, use_tanh=False)
        self.outlayer = nn.Softmax()
        self.recurence = nn.LSTMCell(out_dim)
        self.mask = None

    def forward(self, inputs, embbed_inputs, hidden, encoder_outputs):
    
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)  # or until terminating symbol ?
        inputs = []
        idxs = None
        mask = None
       
        for i in steps:
            output, hidden = self.recurence(decoder_input, hidden)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(output, encoder_outputs)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2) 
            _, logits = self.pointer(g_l, encoder_outputs)
            
            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            decoder_input, idxs = self.decode(
                probs,
                embedded_inputs,
                selections)
            inputs.append(decoder_input) 
        
            outputs.append(probs)
            selections.append(idxs)
        return (outputs, selections), hidden

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):    
        if mask is None:
            mask = torch.zeros(logits.size()).byte()
            if self.use_cuda:
                mask = mask.cuda()
    
        maskk = mask.clone()
        if prev_idxs is not None:
            maskk[[x for x in range(logits.size(0))],
                    prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def decode(self, probs, embedded_inputs, selections):
        batch_size = probs.size(0)
        idxs = probs.multinomial().squeeze(1)

        for old_idxs in selections:
            if old_idxs.eq(idxs).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
                break

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
        return sels, idxs


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


class PointerNetwork(nn.Module):
    def __init__(self):
        super(PointerNetwork, self).__init__()