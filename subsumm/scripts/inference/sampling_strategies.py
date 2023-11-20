import sys
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from subsumm.utils.helpers.data import chunk_data
from shared_lib.utils.helpers.io import read_data
from shared_lib.utils.helpers.logging_funcs import init_logger
from torch.cuda import empty_cache
from shared_lib.utils.constants.general import SEQ_SEP
from argparse import ArgumentParser
from fairseq.models.roberta import RobertaModel
import os
import random
import numpy as np


def random_sampling(data_path, split='valid',
                    output_folder_path='outputs/p',
                    ndocs=10, batch_size=10):
    """Samples reviews randomly."""
    # paths
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    src_out_file_path = os.path.join(output_folder_path, f'{split}.source')
    src = read_data(src_inp_file_path)

    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    safe_mkfdir(src_out_file_path)
    out_src_file = open(src_out_file_path, mode='w', encoding='utf-8')

    logger.info(f"Sampling {ndocs} reviews RANDOMLY for each data point "
                f"from {src_inp_file_path}.")

    for src_chunk in src_chunks:
        # TODO: please note that the number of tags can be different from the
        # TODO: number of initial reviews because of filtering on the
        # TODO: dataset side
        docs = []
        for sample in src_chunk:
            revs = sample.split(f'{SEQ_SEP} ')
            for i in range(16):
                if len(revs) > ndocs:
                    n_revs = random.sample(revs, ndocs)
                    docs.append(f'{SEQ_SEP} '.join(n_revs))
                else:
                    docs.append(sample)

        for _docs in docs:
            out_src_file.write(_docs + '\n')
            out_src_file.flush()

        count += 1
        chunk_process_count += len(src_chunk)
        empty_cache()
        if print_period > 0 and (count % print_period == 0):
            logger.info(f"Processed {chunk_process_count}/{len(src)} lines")

    out_src_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


def sentiment_random_sampling(data_path, dataset, target,
                      split='valid', output_folder_path='outputs/p',
                      ndocs=10, batch_size=10):
    """Samples reviews depending on their sentiments."""
    # paths
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    tag_inp_file_path = os.path.join(data_path, f'{split}.senti')
    src_out_file_path = os.path.join(output_folder_path, f'{split}.source')
    src = [(s, t) for s, t in zip(read_data(src_inp_file_path), read_data(tag_inp_file_path))]

    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    safe_mkfdir(src_out_file_path)
    out_src_file = open(src_out_file_path, mode='w', encoding='utf-8')

    logger.info(f"Sampling {ndocs} reviews depending on the SENTIMENT TAG for each data point "
                f"from {src_inp_file_path}.")

    for src_chunk in src_chunks:
        # TODO: please note that the number of tags can be different from the
        # TODO: number of initial reviews because of filtering on the
        # TODO: dataset side
        docs = []
        for sample in src_chunk:
            revs = [r.strip() for r in sample[0].split(f'{SEQ_SEP} ')]
            sens = sample[1].split()
            assert len(revs) == len(sens)
            if len(revs) > ndocs:
                if dataset == 'amasum':
                    if target == 'cons':
                        poss = [revs[i] for i in range(len(revs)) if float(sens[i]) > 3.0]
                        negs = [revs[i] for i in range(len(revs)) if float(sens[i]) <= 3.0]
                        if len(negs) >= ndocs:
                            revs = random.sample(negs, ndocs)
                        else:
                            revs = negs + random.sample(poss, ndocs - len(negs))
                            random.shuffle(revs)
                    elif target == 'pros':
                        poss = [revs[i] for i in range(len(revs)) if float(sens[i]) >= 3.0]
                        negs = [revs[i] for i in range(len(revs)) if float(sens[i]) < 3.0]
                        if len(poss) >= ndocs:
                            revs = random.sample(poss, ndocs)
                        else:
                            revs = poss + random.sample(negs, ndocs - len(poss))
                            random.shuffle(revs)
                    elif target == 'verd':
                        poss = [revs[i] for i in range(len(revs)) if float(sens[i]) >= 3.0]
                        negs = [revs[i] for i in range(len(revs)) if float(sens[i]) < 3.0]
                        n_pos = int(ndocs * len(poss) / len(revs))
                        revs = random.sample(poss, n_pos) + random.sample(negs, ndocs - n_pos)
                        random.shuffle(revs)

                elif dataset == 'rotten':
                    poss = [revs[i] for i in range(len(revs)) if sens[i] != '0']
                    negs = [revs[i] for i in range(len(revs)) if sens[i] == '0']
                    n_pos = int(ndocs * len(poss) / len(revs))
                    revs = random.sample(poss, n_pos) + random.sample(negs, ndocs - n_pos)
                    random.shuffle(revs)
                docs.append(f'{SEQ_SEP} '.join(revs))
            else:
                docs.append(sample[0])

        for _docs in docs:
            out_src_file.write(_docs + '\n')
            out_src_file.flush()

        count += 1
        chunk_process_count += len(src_chunk)
        empty_cache()
        if print_period > 0 and (count % print_period == 0):
            logger.info(f"Processed {chunk_process_count}/{len(src)} lines")

    out_src_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


def sentiment_information_ranking(data_path, dataset, target,
                      split='valid', output_folder_path='outputs/p',
                      ndocs=10, batch_size=10):
    """Selects reviews depending on their sentiments and information value rankings."""
    # paths
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    tag_inp_file_path = os.path.join(data_path, f'{split}.senti')
    rank_inp_file_path = os.path.join(data_path, f'{split}.rank')
    src_out_file_path = os.path.join(output_folder_path, f'{split}.source')
    src = [(s, t, r) for s, t, r in zip(read_data(src_inp_file_path), read_data(tag_inp_file_path), read_data(rank_inp_file_path))]

    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    safe_mkfdir(src_out_file_path)
    out_src_file = open(src_out_file_path, mode='w', encoding='utf-8')

    logger.info(f"Selecting {ndocs} reviews depending on the SENTIMENT TAG and INFORMATION VALUE for each data point "
                f"from {src_inp_file_path}.")

    for src_chunk in src_chunks:
        # TODO: please note that the number of tags can be different from the
        # TODO: number of initial reviews because of filtering on the
        # TODO: dataset side
        docs = []
        for sample in src_chunk:
            revs = sample[0].split(f'{SEQ_SEP} ')
            sens = sample[1].split()
            rank = eval(sample[2])
            if len(revs) > ndocs:
                assert len(revs) == len(sens)
                #assert len(revs) == len(rank)

                if dataset == 'amasum':
                    if target == 'cons':
                        poss = [i for i in range(len(revs)) if float(sens[i]) > 3.0]
                        negs = [i for i in range(len(revs)) if float(sens[i]) <= 3.0]
                        if len(negs) >= ndocs:
                            idx = [i for i in rank if i in negs][:ndocs]
                        else:
                            idx = negs + [i for i in rank if i in poss][:ndocs - len(negs)]
                            random.shuffle(idx)
                        revs = [revs[i] for i in idx]

                    elif target == 'pros':
                        poss = [i for i in range(len(revs)) if float(sens[i]) >= 3.0]
                        negs = [i for i in range(len(revs)) if float(sens[i]) < 3.0]
                        if len(poss) >= ndocs:
                            idx = [i for i in rank if i in poss][:ndocs]
                        else:
                            idx = poss + [i for i in rank if i in negs][:ndocs - len(poss)]
                            random.shuffle(idx)
                        revs = [revs[i] for i in idx]

                    elif target == 'verd':
                        poss = [i for i in range(len(revs)) if float(sens[i]) >= 3.0]
                        negs = [i for i in range(len(revs)) if float(sens[i]) < 3.0]
                        n_pos = int(ndocs * len(poss) / len(revs))
                        idx = [i for i in rank if i in poss][:n_pos] + [i for i in rank if i in negs][:ndocs - n_pos]
                        random.shuffle(idx)
                        revs = [revs[i] for i in idx]

                elif dataset == 'rotten':
                    poss = [i for i in range(len(revs)) if sens[i] != '0']
                    negs = [i for i in range(len(revs)) if sens[i] == '0']
                    n_pos = int(ndocs * len(poss) / len(revs))
                    idx = [i for i in rank if i in poss][:n_pos] + [i for i in rank if i in negs][:ndocs - n_pos]
                    random.shuffle(idx)
                    revs = [revs[i] for i in idx]
                docs.append(f'{SEQ_SEP} '.join(revs))
            else:
                docs.append(sample[0])

        for _docs in docs:
            out_src_file.write(_docs + '\n')
            out_src_file.flush()

        count += 1
        chunk_process_count += len(src_chunk)
        empty_cache()
        if print_period > 0 and (count % print_period == 0):
            logger.info(f"Processed {chunk_process_count}/{len(src)} lines")

    out_src_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True,
                        help="Location of FAIRSEQ (not binarized) data")
    parser.add_argument('--dataset', type=str, default='amasum', choices=['amasum', 'rotten'])
    parser.add_argument('--target', type=str, default='cons', choices=['pros', 'cons', 'verd'])
    parser.add_argument('--split', default='valid')
    parser.add_argument('--output-folder-path', required=True,
                        default='output/p')
    parser.add_argument('--ndocs', type=int, default=10,
                        help="The number of documents to select")
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--select-strategy', required=True, type=str, choices=['r', 'sr', 'si'], default='r')
    args = parser.parse_args()

    if args.select_strategy == 'r':
        random_sampling(args.data_path, args.split, args.output_folder_path, args.ndocs, args.batch_size)
    elif args.select_strategy == 'sr':
        sentiment_random_sampling(args.data_path, args.dataset, args.target, args.split, args.output_folder_path, args.ndocs, args.batch_size)
    elif args.select_strategy == 'si':
        sentiment_information_ranking(args.data_path, args.dataset, args.target, args.split, args.output_folder_path, args.ndocs, args.batch_size)
