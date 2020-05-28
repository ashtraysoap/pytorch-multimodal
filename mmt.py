# nacitaj data do nejakeho iteratoru

# iterator vracia data [max_seq_len, batch, 1]

# strci sa do modelu:

# embedding -> [max_seq_len, batch, emb_dim]

# h_0 : [num_layers * num_directions, batch, hid_dim]

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from attentions import AdditiveAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pdb
class TextEncoder(nn.Module):

    def __init__(self, src_dim, hid_dim, emb_dim, max_seq_len):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(src_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, bidirectional=True)

    def forward(self, input):
        # input: [batch, seq_len, 1]
        o = self.embedding(input)
        o = o.squeeze(dim=2)
        o = o.permute(1, 0, 2)
        o, h = self.gru(o)
        # o : [seq_len, batch, 2 * hid_dim]
        return o, h

class MMTDecoder(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        src_cap_ctx_dim, 
        img_ctx_dim, 
        img_attn_dim, 
        src_cap_attn_dim, 
        dropout_p=0.1):
        
        super(MMTDecoder, self).__init__()

        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.img_attn = AdditiveAttention(dim_q=hid_dim, dim_k=img_ctx_dim, hid_dim=img_attn_dim)
        self.cap_attn = AdditiveAttention(dim_q=hid_dim, dim_k=src_cap_ctx_dim, hid_dim=src_cap_attn_dim)
        self.attn_combine = nn.Linear(emb_dim + src_cap_ctx_dim + img_ctx_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.hid_dim = hid_dim



    def forward(self, input, hidden, caption_encoder_outputs, image_encoder_outputs):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            caption_encdoer_outputs: [batch, max_seq_len, src_cap_ctx_dim]
            image_encoder_outputs: [batch, n_keys_i, img_ctx_dim]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)

        # Image attention
        ctx_img, a = self.img_attn(Q=hidden.squeeze(0), 
                    K=image_encoder_outputs, 
                    V=image_encoder_outputs)

        # Caption attention
        ctx_cap, _ = self.cap_attn(Q=hidden.squeeze(0),
                    K=caption_encoder_outputs,
                    V=caption_encoder_outputs)

        o = torch.cat((o[0], ctx_img, ctx_cap), dim=1)

        o = self.attn_combine(o)

        o = o.unsqueeze(0)
        o = F.relu(o)

        o, h = self.gru(o, hidden)

        #out = self.out(y=emb, h=hid, z=context)
        o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class MMTNetwork(nn.Module):

    def __init__(self, 
        src_emb_dim, 
        tgt_emb_dim, 
        enc_dim, 
        dec_dim, 
        src_dim,
        out_dim, 
        img_attn_dim, 
        src_cap_attn_dim,
        sos_token=0,
        eos_token=1,
        pad_token=2,
        max_seq_len=20, 
        teacher_forcing_rat=0.90,
        dropout_p=0.1):

        super(MMTNetwork, self).__init__()

        self.encoder = TextEncoder(src_dim=src_dim, hid_dim=enc_dim, emb_dim=src_emb_dim, max_seq_len=max_seq_len)
        self.decoder = MMTDecoder(hid_dim=dec_dim, emb_dim=tgt_emb_dim, out_dim=out_dim, 
            src_cap_ctx_dim=2 * enc_dim, img_ctx_dim=512, img_attn_dim=img_attn_dim, 
            src_cap_attn_dim=src_cap_attn_dim, dropout_p=dropout_p)

        self.out_dim = out_dim
        self.max_seq_len = max_seq_len
        self.teacher_forcing_rat = teacher_forcing_rat
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def forward(self, source_captions, image_features, target_captions, **kwargs):
        """
        Shapes:
            source_captions: [batch_size, max_input_len, 1]
            image_features: [batch_size, X, Y]
            target_captions: [max_len, batch_size, 1]
        """

        max_len = self.max_seq_len
        # features : [batch, enc_seq_len, enc_dim]
        batch_size = image_features.size()[0]
        
        y = torch.tensor([[self.sos_token]] * batch_size, device=device).view(1, batch_size, 1)
        hid = self.decoder.initHidden(batch_size=batch_size)
        
        # gradually store outputs here:
        outputs = torch.zeros(max_len, batch_size, self.out_dim, device=device)
        attentions = torch.zeros(max_len, batch_size, 196, device=device)

        # Encode source captions
        srcs_encoded, _ = self.encoder(source_captions)
        #pdb.set_trace()
        for i in range(self.max_seq_len):
            out, hid, att = self.decoder(y, hid, srcs_encoded, image_features)
            outputs[i] = out.squeeze(dim=0)
            attentions[i] = att

            _, topi = out.topk(1)

            if random.random() < self.teacher_forcing_rat \
                and target_captions is not None:
                y = target_captions[i].unsqueeze(0) # teacher force
            else:
                y = topi.detach()
        
        return outputs, attentions # output logits in shape [max_len, batch, vocab]