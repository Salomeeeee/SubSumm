import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.bart.model import bart_base_architecture
from fairseq import checkpoint_utils
from subsumm.utils.helpers.model import update_names
from subsumm.interfaces.ranker_interface import RankerInterface
import logging


logger = logging.getLogger(__name__)


@register_model('review_ranker')
class ReviewRanker(TransformerModel):
    """An Transformer Encoder."""

    def __init__(self, args, encoder, decoder):
        super(ReviewRanker, self).__init__(args, encoder, decoder)
        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, 
                return_all_hiddens: bool = False):
        enc_out = self.encoder(src_tokens=src_tokens,
                               src_lengths=src_lengths,
                               return_all_hiddens=return_all_hiddens)
        corr = self.compute_correlation(enc_out)

        return enc_out, corr

    def compute_correlation(self, enc_out_obj):
        enc_out = enc_out_obj.encoder_out   # [src_len, sample_size, embed_dim]
        enc_pad_mask = enc_out_obj.encoder_padding_mask # [sample_size, src_len]
        mask = (~enc_pad_mask).transpose(0, 1).unsqueeze(-1).expand(-1, -1, enc_out.size(2))    # [src_len, sample_size, embed_dim]

        rev_embs = torch.sum(enc_out * mask, 0) / torch.sum(mask, 0)  # [sample_size, embed_dim]
        leave_one_avg = (torch.sum(rev_embs, 0) - rev_embs) / (rev_embs.size(0) - 1)
        # corr = torch.nn.functional.cosine_similarity(rev_embs, leave_one_avg)
        corr = torch.sum(rev_embs * leave_one_avg, 1)

        return corr

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)
        # return None

    @classmethod
    def from_pretrained(cls, checkpoint_file, **kwargs):
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_file], arg_overrides=kwargs)
        return RankerInterface(args, task, models[0])

    def upgrade_state_dict_named(self, state_dict, name):
        update_names(self, name, state_dict)


@register_model_architecture('review_ranker', 'ranker_base')
def ranker_base_architecture(args):
    bart_base_architecture(args)
