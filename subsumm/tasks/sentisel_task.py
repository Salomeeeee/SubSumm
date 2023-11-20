import logging
from macpath import split
import os
import torch as T
import numpy as np
import random
from selsum.tasks.abs_task import AbsTask, MAX_DOC_LEN
from selsum.data.abs_dataset import AbsDataset
from selsum.data.abs_dataset import AbsDataset
from selsum.data.doc_reducer import DocReducer
from selsum.data.seq_splitter import SeqSplitter
from selsum.data.seq_dataset import SeqDataset
from selsum.data.src_wrapper import SrcWrapper
from selsum.utils.helpers.io import make_bin_path
from fairseq.data import BaseWrapperDataset, StripTokenDataset
from fairseq.data.data_utils import load_indexed_dataset
from selsum.utils.constants.model import SEP_REPL
from fairseq.tasks import FairseqTask, register_task
from selsum.utils.helpers.model import setup_bpe


logger = logging.getLogger(__name__)


@register_task('sentisel_task')
class SentiSelTask(AbsTask):
    """First a subset of reviews is selected depending on the sentiment tags and subsequently these reviews are summarized.
    
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        AbsTask.add_args(parser)
        parser.add_argument('--ndocs', type=int, default=10,
                            help='The maximum number of reviews/documents to '
                                 'sample per instance.')      
        parser.add_argument('--dataset-name', type=str, choices=['amasum', 'rotten'])                          
        parser.add_argument('--target-name', type=str, choices=['pros', 'cons', 'verd'])                          


    def __init__(self, args, dictionary):
        super(SentiSelTask, self).__init__(args, dictionary)
        self.ndocs = args.ndocs
        self.dataset_name = args.dataset_name
        self.target_name = args.target_name

    def load_dataset(self, split, epoch=1, **kwargs):
        # paths
        base_path = self.args.data
        bin_path = make_bin_path(base_path)
        src_path = os.path.join(bin_path, f"{split}.source-target.source")
        tgt_path = os.path.join(bin_path, f"{split}.source-target.target")

        # loading
        with open(os.path.join(base_path, f'{split}.senti'), 'r', encoding='utf-8') as f:
            tags = [tag_str.strip().split() for tag_str in f.readlines()]
        logger.info("Split: {0}, Loaded {1} tag strings".format(split, len(tags)))

        src_ds = load_indexed_dataset(src_path, dictionary=self.dictionary)
        if src_ds is None:
            raise ValueError(f"Could not load the source dataset in "
                             f"'{src_path}'.")
        src_ds = self._create_source_dataset(dataset=src_ds,
                                             tags = tags,
                                             max_doc_len=MAX_DOC_LEN,
                                             dataset_sizes=src_ds.sizes)
        tgt_ds = self._load_target_dataset(tgt_path)

        ds = AbsDataset(src=src_ds, tgt=tgt_ds,
                        shuffle=self.args.shuffle,
                        pad_indx=self.dictionary.pad(),
                        bos_indx=self.dictionary.bos(),
                        eos_indx=self.dictionary.eos(),
                        mask_indx=self.mask_indx)
        logger.info("Split: {0}, Loaded {1} samples".format(split, len(ds)))
        logger.info(f"The dataset size: {len(ds)}")
        self.datasets[split] = ds

    def _create_source_dataset(self, dataset, tags, dataset_sizes, max_budgets=None,
                               max_doc_len=None):
        """Properly wraps the source dataset. If no features are provided,
        will simply split sub-sequences.
        """
        dataset = SeqDataset(dataset, dataset_sizes)
        dataset = StripTokenDataset(dataset, self.dictionary.eos())
        dataset = SeqSplitter(dataset=dataset, sep_indxs=self.sep_indxs)
        dataset = SentiSelWrapper(dataset=dataset, ndocs=self.ndocs, tags=tags, 
                                  dataset_name=self.dataset_name, target_name=self.target_name)
        if max_budgets is not None or max_doc_len is not None:
            dataset = DocReducer(dataset=dataset, max_budgets=max_budgets,
                                 sort_docs=False, max_doc_len=max_doc_len)
        dataset = SrcWrapper(dataset=dataset, bos_indx=self.dictionary.bos(),
                             eos_indx=self.dictionary.eos())

        return dataset


class SentiSelWrapper(BaseWrapperDataset):

    def __init__(self, dataset, ndocs, tags, dataset_name='amasum', target_name='pros'):
        super(SentiSelWrapper, self).__init__(dataset)
        self.ndocs = ndocs
        self.tags = tags
        self.dataset_name = dataset_name
        self.target_name = target_name

    def __getitem__(self, index):
        src_docs = self.dataset[index]
        if len(src_docs) > self.ndocs:
            sens = self.tags[index]
            assert len(sens) == len(src_docs), f'{len(sens)} tags VS {len(src_docs)} in sample No. {index}'

            if self.dataset_name == 'amasum':
                if self.target_name == 'cons':
                    poss = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) > 3.0]
                    negs = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) <= 3.0]
                    if len(negs) >= self.ndocs:
                        src_docs = random.sample(negs, self.ndocs)
                    else:
                        src_docs = negs + random.sample(poss, self.ndocs - len(negs))
                        random.shuffle(src_docs)
                elif self.target_name == 'pros':
                    poss = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) >= 3.0]
                    negs = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) < 3.0]
                    if len(poss) >= self.ndocs:
                        src_docs = random.sample(poss, self.ndocs)
                    else:
                        src_docs = poss + random.sample(negs, self.ndocs - len(poss))
                        random.shuffle(src_docs)
                elif self.target_name == 'verd':
                    poss = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) >= 3.0]
                    negs = [src_docs[i] for i in range(len(src_docs)) if float(sens[i]) < 3.0]
                    n_pos = int(self.ndocs * len(poss) / len(src_docs))
                    src_docs = random.sample(poss, n_pos) + random.sample(negs, self.ndocs - n_pos)
                    random.shuffle(src_docs)
            elif self.dataset_name == 'rotten':
                poss = [src_docs[i] for i in range(len(src_docs)) if sens[i] != '0']
                negs = [src_docs[i] for i in range(len(src_docs)) if sens[i] == '0']
                n_pos = int(self.ndocs * len(poss) / len(src_docs))
                src_docs = random.sample(poss, n_pos) + random.sample(negs, self.ndocs - n_pos)
                random.shuffle(src_docs)
        return src_docs

