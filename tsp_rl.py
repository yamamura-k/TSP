from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
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
    def __init__(self, emb_dim, out_dim, max_len, C, tanh_flg, n_glimpses=1):
        super(Decoder, self).__init__()
        
        self.embedding_dim = emb_dim
        self.hidden_dim = out_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_len

        self.input_weights = nn.Linear(emb_dim, 4 * out_dim)
        self.out_weights = nn.Linear(out_dim, 4 * out_dim)

        self.pointer = Attention(out_dim, tanh_flg=tanh_flg, C=C)
        self.glimpse = Attention(out_dim, tanh_flg=False)
        self.outlayer = nn.Softmax()
        self.mask = None

    def recurrence(self, x, hidden):
        hx, cx = hidden  # batch_size x hidden_dim
        gates = self.input_weights(x) + self.out_weights(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy
            
    def forward(self, decoder_input, embbed_inputs, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)
        inputs = []
        idxs = None
        mask = None
       
        for step in steps:
            (output, cy) = self.recurrence(decoder_input, hidden)
            hidden = (output, cy)
            g_l = output
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(output, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(step, logits, mask, idxs)
                g_l = torch.bmm(ref, self.outlayer(logits).unsqueeze(2)).squeeze(2) 
            _, logits = self.pointer(g_l, encoder_outputs)
            
            logits, mask = self.apply_mask_to_logits(step, logits, mask, idxs)
            probs = self.outlayer(logits)
            decoder_input, idxs = self.decode(
                probs,
                embbed_inputs,
                selections)
            inputs.append(decoder_input) 
        
            outputs.append(probs)
            selections.append(idxs)
        return (outputs, selections), hidden

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):    
        if mask is None:
            mask = torch.zeros(logits.size()).byte()
    
        maskk = mask.clone()
        if prev_idxs is not None:
            maskk[[x for x in range(logits.size(0))],
                    prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def decode(self, probs, embedded_inputs, selections):
        batch_size = probs.size(0)
        try:
            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in selections:
                if old_idxs.eq(idxs).data.any():
                    print(' [!] resampling due to race condition')
                    idxs = probs.multinomial(batch_size).squeeze(1)
                    break
        except:
            idxs = selections[-1]

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
        return sels, idxs


class Attention(nn.Module):
    """base class of pointer & glimpse
    """
    def __init__(self, layer_dim, C=10, tanh_flg=False):
        super(Attention, self).__init__()
        self.proj_q = nn.Linear(layer_dim, layer_dim)# hidden vector of decoder
        #self.proj_ref = nn.Linear(layer_dim, layer_dim)# hidden vector of encoder
        self.C = C
        self.tanh = nn.Tanh()
        self.tanh_flg = tanh_flg
        self.proj_ref = nn.Conv1d(layer_dim, layer_dim, 1, 1)
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
        logits = self.C*self.tanh(u) if self.tanh_flg else u
        return e, logits


class PointerNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, max_len, n_glimpses, C, tanh_flg):
        super(PointerNetwork, self).__init__()
        self.encoder = Encoder(emb_dim, hidden_dim)
        self.decoder = Decoder(emb_dim, hidden_dim, max_len, C, tanh_flg, n_glimpses=n_glimpses)

        dec_in_init = torch.FloatTensor(emb_dim)
        
        self.decoder_in_0 = nn.Parameter(dec_in_init)
        self.decoder_in_0.data.uniform_(-(1./sqrt(emb_dim)), 1./sqrt(emb_dim))
    
    def forward(self, inputs):
        enc_h, enc_c = self.encoder.enc
        enc_h = enc_h.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        enc_c = enc_c.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)       
        
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (enc_h, enc_c))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)
        
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input, inputs, dec_init_state, enc_h)

        return pointer_probs, input_idxs

class NCombOptRL(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, max_len, n_glimpses, n_process_block_iter, C, tanh_flg, obj_func, is_trained):
        super(NCombOptRL, self).__init__()
        self.obj_func = obj_func
        self.input_dim = input_dim
        self.is_trained = is_trained

        self.actor = PointerNetwork(emb_dim, hidden_dim, max_len, n_glimpses, C, tanh_flg)
        emb = torch.FloatTensor(input_dim, emb_dim)
        self.emb = nn.Parameter(emb)
        self.emb.data.uniform_(-(1./sqrt(emb_dim)), 1./sqrt(emb_dim))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        source = inputs.size(2)
        emb = self.emb.repeat(batch_size, 1, 1)  
        emb_inputs = []
        ips = inputs.unsqueeze(1)
        print(inputs.size())
        print(ips.size(), emb.size())
        
        for i in range(source):
            emb_inputs.append(torch.bmm(ips[:, :, :, i].float(), emb).squeeze(1))

        emb_inputs = torch.cat(emb_inputs).view(source, batch_size, emb.size(2))

        probs_, action_idxs = self.actor(emb_inputs)

        actions = []
        inputs_ = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs_[[x for x in range(batch_size)], action_id.data, :])

        if self.is_trained:
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[[x for x in range(batch_size)], action_id.data])
        else:
            probs = probs_

        Rewards = self.obj_func(actions)
        
        return Rewards, probs, actions, action_idxs