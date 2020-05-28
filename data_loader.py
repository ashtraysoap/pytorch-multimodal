"""A class for loading and serving input data for the image captioning task.
"""

import os
import random

import numpy as np
import torch

class DataLoader:
    
    def __init__(
        self, 
        captions_src,
        captions_tgt, 
        sources, 
        batch_size=1, 
        sources_prefix="", 
        vocab_src=None,
        vocab_tgt=None,
        max_seq_len=None,
        shuffle=True
    ):
        
        captions_src = list(captions_src)
        captions_tgt = list(captions_tgt)
        sources = list(sources)

        if sources_prefix != "":
            sources = [os.path.join(sources_prefix, s) for s in sources]
        
        assert len(sources) == len(captions_src) == len(captions_tgt)
        
        self.sources = sources
        self.captions_src = captions_src
        self.captions_tgt = captions_tgt
        self.batch_size = batch_size
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self._count = len(sources)

    def __iter__(self):
        if self.shuffle == True:
            pairs = list(zip(self.captions_src, self.sources, self.captions_tgt))
            random.shuffle(pairs)
            sources, captions_src, captions_tgt = [], [], []
            for c_s, s, c_t in pairs:
                sources.append(s)
                captions_src.append(c_s)
                captions_tgt.append(c_t)
        else:
            sources, captions_src, captions_tgt = self.sources, self.captions_src, captions_tgt

        counter = 0
        while counter < self._count:
            upper_bound = min(self._count, counter + self.batch_size)
            srcs = sources[counter : upper_bound]
            xs = [_load_input_data(s) for s in srcs]
            xs = [_reshape(x) for x in xs]
            xs = [torch.from_numpy(x) for x in xs]
            xs = torch.stack(xs, dim=0)
            caps_in = captions_src[counter : upper_bound]

            caps_out = captions_tgt[counter : upper_bound]
            num_instances = len(xs)
            if self.vocab_src is not None:
                caps_in = [self.vocab_src.sentence_to_tensor(y, self.max_seq_len) for y in caps_in]
                caps_in = torch.stack(caps_in, dim=0)
                caps_in = caps_in.permute(1, 0, 2)

            if self.vocab_tgt is not None:
                caps_out = [self.vocab_tgt.sentence_to_tensor(y, self.max_seq_len) for y in caps_out]
                caps_out = torch.stack(caps_out, dim=0)
                caps_out = caps_out.permute(1, 0, 2)
            
            yield (caps_in, xs, caps_out, num_instances)
            counter = upper_bound


def _load_input_data(source):
    """Load the data into memory from the given path.
    """
    if source[-4:] == ".npz":
        return np.load(source)['arr_0']

    print(source)
    raise NotImplementedError("Only .npz support so far.")

def _reshape(x):
    """x is a numpy array with assumed shape (w, h, dim) or (s, dim).
    """
    if len(x.shape) == 2:
        return x
    if len(x.shape) == 3:
        dim = x.shape[2]
        return np.reshape(x, (-1, dim))
    
    raise NotImplementedError("Incorrectly shaped array given.")
