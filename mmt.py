import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from attentions import AdditiveAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AttentionDecoder_1(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        img_ctx_dim,
        src_cap_ctx_dim,
        img_attn_dim,
        src_cap_attn_dim,
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_1, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.img_attention = attention(dim_k=img_ctx_dim, dim_q=hid_dim, hid_dim=img_attn_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.cap_attention = attention(dim_k=src_cap_ctx_dim, dim_q=hid_dim, hid_dim=src_cap_attn_dim,
                dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.gru = nn.GRU(emb_dim + src_cap_ctx_dim + img_ctx_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, caption_encoder_outputs, image_encoder_outputs):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        emb = o

        ctx_img, a = self.img_attention(Q=hidden.squeeze(0), 
                    K=image_encoder_outputs, 
                    V=image_encoder_outputs)
        
        ctx_cap, _ = self.cap_attention(Q=hidden.squeeze(0),
                    K=caption_encoder_outputs,
                    V=caption_encoder_outputs)

        o = torch.cat((o[0], ctx_img, ctx_cap), dim=1)
        o = o.unsqueeze(0)

        o, h = self.gru(o, hidden)

        if self.deep_out:
            o = self.out(y=emb, h=o, z=context)
        else:
            o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder_2(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        img_ctx_dim,
        src_cap_ctx_dim,
        img_attn_dim,
        src_cap_attn_dim,
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_2, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.img_attention = attention(dim_k=img_ctx_dim, dim_q=hid_dim, hid_dim=img_attn_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.cap_attention = attention(dim_k=src_cap_ctx_dim, dim_q=hid_dim, hid_dim=src_cap_attn_dim,
                dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.attn_combine = nn.Linear(emb_dim + src_cap_ctx_dim + img_ctx_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, caption_encoder_outputs, image_encoder_outputs):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        emb = o

        ctx_img, a = self.img_attention(Q=hidden.squeeze(0), 
                    K=image_encoder_outputs, 
                    V=image_encoder_outputs)

        ctx_cap, _ = self.cap_attention(Q=hidden.squeeze(0),
                    K=caption_encoder_outputs,
                    V=caption_encoder_outputs)

        o = torch.cat((o[0], ctx_img, ctx_cap), dim=1)

        o = self.attn_combine(o)

        o = o.unsqueeze(0)
        o = F.relu(o)

        o, h = self.gru(o, hidden)

        if self.deep_out:
            o = self.out(y=emb, h=o, z=context)
        else:
            o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder_3(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        img_ctx_dim,
        src_cap_ctx_dim,
        img_attn_dim,
        src_cap_attn_dim,
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_3, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.img_attention = attention(dim_q=hid_dim, 
            dim_k=img_ctx_dim, hid_dim=img_attn_dim,
            dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.cap_attention = attention(dim_q=hid_dim, 
            dim_k=src_cap_ctx_dim, hid_dim=src_cap_attn_dim,
            dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.attn_combine = nn.Linear(hid_dim + src_cap_ctx_dim + img_ctx_dim, hid_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, caption_encoder_outputs, image_encoder_outputs):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        emb = o

        _, h_t = self.gru(o, hidden)

        ctx_img, a_t = self.img_attention(Q=h_t.squeeze(0), 
                    K=image_encoder_outputs, 
                    V=image_encoder_outputs)

        ctx_cap, _ = self.cap_attention(Q=h_t.squeeze(0), 
                    K=caption_encoder_outputs, 
                    V=caption_encoder_outputs)


        o_t = torch.cat((h_t.squeeze(0), ctx_img, ctx_cap), dim=1)

        o_t = self.attn_combine(o_t)
        o_t = o_t.unsqueeze(0)
        o_t = F.relu(o_t)

        if self.deep_out:
            out = self.out(y=emb, h=o_t, z=torch.cat((ctx_img, ctx_cap), dim=1))
        else:
            out = self.out(o_t)
        logits = self.log_softmax(out)

        return logits, h_t, a_t

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)


class AttentionDecoder_4(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        img_ctx_dim,
        src_cap_ctx_dim,
        img_attn_dim,
        src_cap_attn_dim,
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_4, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.img_attention = attention(dim_q=hid_dim, 
            dim_k=img_ctx_dim, hid_dim=img_attn_dim,
            dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.cap_attention = attention(dim_q=hid_dim, 
            dim_k=src_cap_ctx_dim, hid_dim=src_cap_attn_dim,
            dropout_p=dropout_p, attn_activation=attn_activation)
        
        self.attn_combine = nn.Linear(hid_dim + src_cap_ctx_dim + img_ctx_dim, hid_dim)
        self.gru = nn.GRU(hid_dim + emb_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, caption_encoder_outputs, image_encoder_outputs):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """
        prev_h, prev_o = hidden

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        emb = o

        in_t = torch.cat((o, prev_o), dim=2)

        _, h_t = self.gru(in_t, prev_h)

        ctx_img, a_t = self.img_attention(Q=h_t.squeeze(0), 
                    K=image_encoder_outputs, 
                    V=image_encoder_outputs)

        ctx_cap, _ = self.cap_attention(Q=h_t.squeeze(0), 
                    K=caption_encoder_outputs, 
                    V=caption_encoder_outputs)


        o_t = torch.cat((h_t.squeeze(0), ctx_img, ctx_cap), dim=1)

        o_t = self.attn_combine(o_t)
        o_t = o_t.unsqueeze(0)
        o_t = F.relu(o_t)

        if self.deep_out:
            out = self.out(y=emb, h=o_t, z=torch.cat((ctx_img, ctx_cap), dim=1))
        else:
            out = self.out(o_t)
        logits = self.log_softmax(out)

        return logits, (h_t, o_t), a_t

    def initHidden(self, batch_size):
        h_0 = torch.zeros(1, batch_size, self.hid_dim, device=device)
        o_0 = torch.zeros(1, batch_size, self.hid_dim, device=device)
        return (h_0, o_0)


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
        dropout_p=0.1,
        deep_out=False,
        attn_activation='relu',
        decoder=AttentionDecoder_1,
        attention=AdditiveAttention):

        super(MMTNetwork, self).__init__()

        print(decoder)
        print(attention)

        self.encoder = TextEncoder(src_dim=src_dim, hid_dim=enc_dim, emb_dim=src_emb_dim, max_seq_len=max_seq_len)
        self.decoder = decoder(hid_dim=dec_dim, emb_dim=tgt_emb_dim, out_dim=out_dim, 
            src_cap_ctx_dim=2 * enc_dim, img_ctx_dim=512, img_attn_dim=img_attn_dim, 
            src_cap_attn_dim=src_cap_attn_dim, dropout_p=dropout_p, attn_activation=attn_activation,
            deep_out=deep_out, attention=attention)

        self.out_dim = out_dim
        self.max_seq_len = max_seq_len
        self.teacher_forcing_rat = teacher_forcing_rat
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.decoder.to(device)
        self.encoder.to(device)

    def forward(self, source_captions, image_features, target_captions=None, max_len=10, teacher_forcing_rat=None, **kwargs):
        """
        Shapes:
            source_captions: [batch_size, max_input_len, 1]
            image_features: [batch_size, X, Y]
            target_captions: [max_len, batch_size, 1]
        """

        if teacher_forcing_rat == None:
            teacher_forcing_rat = self.teacher_forcing_rat

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

        for i in range(self.max_seq_len):
            out, hid, att = self.decoder(y, hid, srcs_encoded, image_features)
            outputs[i] = out.squeeze(dim=0)
            attentions[i] = att

            _, topi = out.topk(1)

            if random.random() < teacher_forcing_rat \
                and target_captions is not None:
                y = target_captions[i].unsqueeze(0) # teacher force
            else:
                y = topi.detach()
        
        return outputs, attentions # output logits in shape [max_len, batch, vocab]

    def infere(self, src_caps, features, max_len=10):
        out, att = self.forward(source_captions=src_caps, 
            image_features=features, 
            target_captions=None, 
            max_len=max_len)
        
        out = out.permute(1, 0, 2).detach()
        att = att.permute(1, 0, 2).detach()

        _, topi = out.topk(1, dim=2)
        topi = topi.squeeze(2)

        return {
            'token_ids': topi,
            'alignments': att,
        }