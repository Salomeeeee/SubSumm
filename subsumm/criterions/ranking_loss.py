from email.policy import default
import torch
import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('ranking_loss')
class RankingLoss(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--margin', type=float, default=0.1)
        parser.add_argument('--ndocs', type=float, default=10)

    def __init__(self, margin, ndocs, task):
        super().__init__(task)
        self.margin = margin
        self.ndocs = ndocs
    
    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        _, corr = model(**sample['net_input'])
        loss = self.compute_loss(corr)

        sample_size = corr.size(0)

        logging_output = {
            'loss': loss.item(),
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, score):
        """ Pair-wise margin ranking loss. """
        n = score.size(0)   # num_docs = sample_size
        TotalLoss = 0
        for i in range(1, n):
            pos_score = score[:-i]
            neg_score = score[i:]
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
        return TotalLoss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

