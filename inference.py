import os

import fire
import numpy as np
import torch
import torch.nn as nn

from vocab import Vocab
import mmt
from mmt import MMTNetwork
import attentions


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MAX_LEN = 15
HIDDEN_DIM = 1024
EMB_DIM = 512
ENC_SEQ_LEN = 14 * 14
ENC_DIM = 512


def run(test_dir, 
    test_srcs,
    test_src_caps,
    checkpoint, 
    vocab_src,
    vocab_tgt, 
    out="captions.out.txt",
    batch_size=16, 
    max_seq_len=MAX_LEN,
    hidden_dim=HIDDEN_DIM,
    emb_dim=EMB_DIM,
    enc_seq_len=ENC_SEQ_LEN,
    enc_dim=ENC_DIM,
    attn_activation="relu",
    deep_out=False,
    decoder=2,
    attention=3):

    if decoder == 1:
        decoder = mmt.AttentionDecoder_1
    elif decoder == 2:
        decoder = mmt.AttentionDecoder_2
    elif decoder == 3:
        decoder = mmt.AttentionDecoder_3
    elif decoder == 4:
        decoder = mmt.AttentionDecoder_4

    if attention == 1:
        attention = attentions.AdditiveAttention
    elif attention == 2:
        attention = attentions.GeneralAttention
    elif attention == 3:
        attention = attentions.ScaledGeneralAttention

    # load vocabulary
    vocabulary_src = Vocab()
    vocabulary_src.load(vocab_src)

    vocabulary_tgt = Vocab()
    vocabulary_tgt.load(vocab_tgt)

    # load test instances file paths
    srcs = open(test_srcs).read().strip().split('\n')
    srcs = [os.path.join(test_dir, s) for s in srcs]

    src_caps = open(test_src_caps, encoding='utf-8').read().strip().split('\n')


    # load model
    net = MMTNetwork(
        src_emb_dim=emb_dim,
        tgt_emb_dim=emb_dim,
        enc_dim=hidden_dim,
        dec_dim=hidden_dim,
        src_dim=vocabulary_src.n_words, 
        out_dim=vocabulary_tgt.n_words,
        img_attn_dim=512,
        src_cap_attn_dim=512,
        sos_token=0, eos_token=1, pad_token=2,
        max_seq_len=max_seq_len,
        deep_out=deep_out,
        attention=attention, decoder=decoder)
    net.to(DEVICE)

    net.load_state_dict(torch.load(checkpoint))
   
    net.eval()

    with torch.no_grad():

        # run inference
        num_instances = len(srcs)

        i = 0
        captions = []
        while i < num_instances:
            srcs_batch = srcs[i:i + batch_size]
            batch = _load_batch(srcs_batch)
            batch = batch.to(DEVICE)

            caps_in = src_caps[i:i + batch_size]
            caps_in = [vocabulary_src.sentence_to_tensor(y, max_seq_len) for y in caps_in]
            caps_in = torch.stack(caps_in, dim=0)
            caps_in = caps_in.permute(1, 0, 2)
            caps_in = caps_in.to(DEVICE)

            tokens, _ = net(source_captions=caps_in,
                image_features=batch, 
                targets=None, 
                max_len=max_seq_len)
            
            tokens = tokens.permute(1, 0, 2).detach()
            _, topi = tokens.topk(1, dim=2)
            topi = topi.squeeze(2)

            # decode token output from the model
            for j in range(len(srcs_batch)):
                c = vocabulary_tgt.tensor_to_sentence(topi[j])
                c = ' '.join(c)
                captions.append(c)

            i += len(srcs_batch)

    out_f = open(out, mode='w', encoding='utf-8')
    for c in captions:
        out_f.write(c + '\n')

    return

def _load_features(fp):
    x = np.load(fp)['arr_0']

    if len(x.shape) == 3:
        dim = x.shape[2]
        x = np.reshape(x, (-1, dim))

    return torch.from_numpy(x)

def _load_batch(fps):
    x = [_load_features(fp) for fp in fps]
    return torch.stack(x, dim=0)


if __name__ == "__main__":
    fire.Fire(run)
