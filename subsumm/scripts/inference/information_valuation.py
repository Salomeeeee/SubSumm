import torch
import sys
from subsumm.models.review_ranker import ReviewRanker
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from subsumm.utils.helpers.data import chunk_data
from shared_lib.utils.helpers.io import read_data
from shared_lib.utils.helpers.logging_funcs import init_logger
from torch.cuda import empty_cache
from shared_lib.utils.constants.general import SEQ_SEP
from argparse import ArgumentParser
import os


def information_valuation(data_path, checkpoint_path, split='valid',
                      output_folder_path='outputs/p', bart_dir='../artifacts/bart/',
                      ndocs=10, batch_size=1):
    """Generate the information value (distribution & rankings) for the review sets."""
    # paths
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    dist_out_file_path = os.path.join(output_folder_path, f'{split}.dist')
    rank_out_file_path = os.path.join(output_folder_path, f'{split}.rank')
    src = read_data(src_inp_file_path)

    imodel = ReviewRanker.from_pretrained(
        bart_dir=bart_dir,
        checkpoint_file=checkpoint_path,
        gpt2_encoder_json=os.path.join(bart_dir, 'encoder.json'),
        gpt2_vocab_bpe=os.path.join(bart_dir, 'vocab.bpe'),
        bpe='gpt2', strict=True
    )

    imodel.cuda()
    imodel.eval()
    imodel.half()

    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    safe_mkfdir(dist_out_file_path)
    out_dist_file = open(dist_out_file_path, mode='w', encoding='utf-8')
    out_rank_file = open(rank_out_file_path, mode='w', encoding='utf-8')

    logger.info(f"Selecting {ndocs} reviews for each data point "
                f"from {src_inp_file_path}.")
    logger.info(f"Inference based on: {checkpoint_path}")

    for src_chunk in src_chunks:
        with torch.no_grad():
            dist, rank, docs = imodel.infer(src_chunk, top_k=ndocs,
                                      out_seq_sep=f'{SEQ_SEP} ')

        out_dist_file.write(str(dist) + '\n')
        out_dist_file.flush()
        out_rank_file.write(str(rank) + '\n')
        out_rank_file.flush()

        count += 1
        chunk_process_count += len(src_chunk)
        empty_cache()
        if print_period > 0 and (count % print_period == 0):
            logger.info(f"Processed {chunk_process_count}/{len(src)} lines")

    out_dist_file.close()
    out_rank_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help="Location of FAIRSEQ (not binarized) data")
    parser.add_argument('--split', default='valid')
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('--output-folder-path', required=True,
                        default='output/p')
    parser.add_argument('--bart-dir', default='../artifacts/bart/')
    parser.add_argument('--ndocs', type=int, default=10,
                        help="The number of documents to select")
    information_valuation(**vars(parser.parse_args()))
