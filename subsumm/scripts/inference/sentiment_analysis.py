from fairseq.models.roberta import RobertaModel
import numpy as np
from tqdm import trange


model_path = ''
src_path = ''   # .source file
out_path = ''   # .senti file
head_name = ''  # film_head for rotten and senti_head for amasum
split = 'test'

roberta = RobertaModel.from_pretrained(model_name_or_path=model_path,
                                       checkpoint_file='checkpoint_best.pt',
                                       data_name_or_path='roberta_dict')
roberta.eval()  # disable dropout
label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])

print(f'Tagging the {split} data...')
with open(src_path, 'r') as f:
    inputs = [revs.strip() for revs in f.readlines()]
fout = open(out_path, 'w')

for i in trange(len(inputs)):
    revs = inputs[i].split(' </s>')
    for rev in revs:
        tokens = roberta.encode(rev)
        pred = label_fn(roberta.predict(head_name, tokens).argmax().item())
        fout.write(f'{pred}\t')
    fout.write('\n')

fout.close()