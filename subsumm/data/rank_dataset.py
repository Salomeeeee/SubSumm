from fairseq.data import FairseqDataset
import numpy as np
import torch
from fairseq.data.data_utils import collate_tokens


def collate(samples, pad_idx, eos_idx):
    '''batch_size must be set to 1.'''

    if len(samples) == 0:
        return {}
    else:
        assert (len(samples) == 1), 'Invalid batch size.'
        s = samples[0]

    idx = torch.LongTensor([s['id']])
    src_tokens = collate_tokens(s['source'], pad_idx, eos_idx,
                                left_pad=False, move_eos_to_beginning=False)
    src_lengths = torch.LongTensor([d.numel() for d in s['source']])
    ntokens = int(sum(src_lengths))

    batch = {
        'id': idx,
        'ntokens': ntokens,
        'nsentences': len(s),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
    }

    return batch


class RankDataset(FairseqDataset):

    def __init__(self, src, pad_indx, bos_indx, eos_indx,
                 shuffle=False):
        super(RankDataset, self).__init__()

        self.src = src
        self.tgt = None
        self.pad_indx = pad_indx
        self.bos_indx = bos_indx
        self.eos_indx = eos_indx

        self.shuffle = shuffle

    def __getitem__(self, index):
        src_docs = self.src[index]
        assert isinstance(src_docs, list)
        return {'id': index, 'source': src_docs}

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples=samples, pad_idx=self.pad_indx,
                       eos_idx=self.eos_indx)
        
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        src_size = self.src.size(index)
        return src_size

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        tgt_size = self.tgt.size(index) if self.tgt else 0
        # src_size = self.src.size(index)
        # TODO: implement the dynamic checking like checking the maximum length
        # TODO: of the source
        return 0, tgt_size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # TODO: make an efficient call for sizes in VI sampler
        # return indices[np.argsort(self.src.sizes[indices], kind='mergesort')]
        return indices

    @property
    def supports_prefetch(self):
        return (getattr(self.src, 'supports_prefetch', False))

    def prefetch(self, indices):
        self.src.prefetch(indices)

