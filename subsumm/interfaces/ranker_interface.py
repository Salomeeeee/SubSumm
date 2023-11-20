from gettext import GNUTranslations
import torch as T
from fairseq.data import encoders
from shared_lib.utils.helpers.topk_ngram_blocker import topk_with_ngram_blocker
from shared_lib.utils.constants.general import SEQ_SEP
import numpy as np
from fairseq import utils
from typing import List
from torch.nn.functional import gumbel_softmax


class RankerInterface(T.nn.Module):
    """Interface for selecting review subsets via the trained prior."""

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        # this is useful for determining the device
        self.register_buffer('_float_tensor', T.tensor([0], dtype=T.float))

    def encode(self, docs):
        """Prepares the documents string for decoding by BPE tokenizing it.

        Args:
            docs (str): concatenated by a separator documents.

        Returns:
            LongTensor with tokens.
        """
        docs = self.bpe.encode(docs)
        tokens = self.task.source_dictionary.encode_line(docs, append_eos=False)
        return tokens.long()

    def decode(self, tokens: T.LongTensor):
        """Converts tokens to the human readable format. Sentence agnostic."""
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        tokens = [t for t in tokens if t not in
                  [self.task.bos_indx, self.task.eos_indx, self.task.pad_indx]]
        res = self.bpe.decode(self.task.source_dictionary.string(tokens))
        res = res.strip()
        return res

    def infer(self, docs, top_k=10, detailed=False, out_seq_sep=f'{SEQ_SEP} '):
        """Ranks documents and selects the top K.

        Args:
            docs (list): document strings. *One string at a time!!!
            top_k (int): that many sequences to select per entry.
            gumbel (bool): whether use gumbel-softmax trick or not.
            ngram_block (int): if passed, will block from being selected n-gram
                overlapping sequences with already selected ones.

        Returns:
            dist (list): the distribution over the review set.
            rank (list): the information value rank.
            sel_doc_coll (str): corresponding documents.
        """
        tokens = [self.encode(_docs) for _docs in docs]
        sample = self._build_sample(tokens)
        src = sample['net_input']['src_tokens']
        out, corr = self.model(**sample['net_input']) # [sample_size,]

        dist = T.nn.functional.softmax(corr, dim=-1)
        rank = T.argsort(corr, descending=True)
        
        # re-creating docs, as some might have been filtered out
        cand_docs = [self.decode(_src) for _src in src]
        sel_doc_coll = out_seq_sep.join([cand_docs[i] for i in rank[:top_k]])

        if detailed:
            return dist.tolist(), rank.tolist(), sel_doc_coll, out
        else:
            return dist.tolist(), rank.tolist(), sel_doc_coll

    def _build_sample(self, src_tokens: List[T.LongTensor], **kwargs):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            **kwargs
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    @property
    def device(self):
        return self._float_tensor.device
