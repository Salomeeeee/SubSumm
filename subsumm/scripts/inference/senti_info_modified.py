import sys
import torch
import os
import random
from subsumm.utils.helpers.io import make_bin_path
from shared_lib.utils.helpers.io import read_data, write_data
from shared_lib.utils.constants.general import SEQ_SEP
from tqdm import tqdm


def sentiment_information_sampling(inp_src_folder_path, split, out_src_folder_path, target=None, n_docs=10, n_samples=16):
    srcs = read_data(os.path.join(inp_src_folder_path, f'{split}.source'))
    tags = read_data(os.path.join(inp_src_folder_path, f'{split}.senti'))
    dists = read_data(os.path.join(inp_src_folder_path, f'{split}.dist'))
    fout = open(os.path.join(out_src_folder_path, f'{split}.source'), 'w', encoding='utf-8')

    for src, tag, dist in tqdm(zip(srcs, tags, dists)):
        revs = src.split(f'{SEQ_SEP} ')
        sens = tag.split()
        dist = eval(dist)

        for j in range(n_samples):
            if len(revs) > n_docs:
                assert len(revs) == len(sens)

                if target == 'cons':
                    poss = [i for i in range(len(revs)) if float(sens[i]) > 3.0]
                    negs = [i for i in range(len(revs)) if float(sens[i]) <= 3.0]
                    if len(dist) != len(revs):  # degenerate into randsel
                        if len(negs) >= n_docs:
                            idx = random.sample(negs, n_docs)
                        else:
                            idx = negs + random.sample(poss, n_docs - len(negs))
                            random.shuffle(idx)
                    else:
                        if len(negs) >= n_docs:
                            idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in negs], num_samples=n_docs, replacement=False))
                            idx = [negs[i] for i in idx]
                        else:
                            pos_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in poss], num_samples=n_docs - len(negs), replacement=False))
                            idx = negs + [poss[i] for i in pos_idx]
                            random.shuffle(idx)

                elif target == 'pros':
                    poss = [i for i in range(len(revs)) if float(sens[i]) >= 3.0]
                    negs = [i for i in range(len(revs)) if float(sens[i]) < 3.0]
                    if len(dist) != len(revs):  # degenerate into randsel
                        if len(poss) >= n_docs:
                            idx = random.sample(poss, n_docs)
                        else:
                            idx = poss + random.sample(negs, n_docs - len(poss))
                            random.shuffle(idx)
                    else:
                        if len(poss) >= n_docs:
                            idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in poss], num_samples=n_docs, replacement=False))
                            idx = [poss[i] for i in idx]
                        else:
                            neg_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in negs], num_samples=n_docs - len(poss), replacement=False))
                            idx = poss + [negs[i] for i in neg_idx]
                            random.shuffle(idx)

                elif target == 'verd':
                    poss = [i for i in range(len(revs)) if float(sens[i]) >= 3.0]
                    negs = [i for i in range(len(revs)) if float(sens[i]) < 3.0]
                    n_pos = int(n_docs * len(poss) / len(revs))
                    if len(dist) != len(revs):  # degenerate into randsel
                        idx = random.sample(poss, n_pos) + random.sample(negs, n_docs - n_pos)
                        random.shuffle(idx)
                    else:
                        try:
                            pos_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in poss], num_samples=n_pos, replacement=False))
                            neg_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in negs], num_samples=n_docs - n_pos, replacement=False))
                            idx = [poss[i] for i in pos_idx] + [negs[i] for i in neg_idx]
                        except:
                            idx = random.sample(poss, n_pos) + random.sample(negs, n_docs - n_pos)
                        random.shuffle(idx)
            
                else:
                    poss = [i for i in range(len(revs)) if sens[i] != '0']
                    negs = [i for i in range(len(revs)) if sens[i] == '0']
                    n_pos = int(n_docs * len(poss) / len(revs))
                    if len(dist) != len(revs):  # degenerate into randsel
                        idx = random.sample(poss, n_pos) + random.sample(negs, n_docs - n_pos)
                        random.shuffle(idx)
                    else:
                        try:
                            pos_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in poss], num_samples=n_pos, replacement=False))
                            neg_idx = list(torch.utils.data.WeightedRandomSampler(weights=[dist[i] for i in negs], num_samples=n_docs - n_pos, replacement=False))
                            idx = [poss[i] for i in pos_idx] + [negs[i] for i in neg_idx]
                        except:
                            idx = random.sample(poss, n_pos) + random.sample(negs, n_docs - n_pos)
                        random.shuffle(idx)

                doc_str = f'{SEQ_SEP} '.join([revs[i] for i in idx])

            else:
                random.shuffle(revs)
                doc_str = f'{SEQ_SEP} '.join(revs)

            fout.write(doc_str + '\n')
            fout.flush()

    fout.close()

    