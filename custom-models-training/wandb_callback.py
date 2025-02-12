from typing import Any, Dict

import numpy as np
from transformers import Trainer
from transformers.integrations import WandbCallback
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput
from transformers.models.electra.modeling_electra import ElectraSelfAttention, ElectraSelfOutput
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from custom_attention import BertSelfAttentionSymmetricInit

from symmetry_utils import get_key_query_matrix, to_symmetric_asymmetric_fast


class WandbSymmetryCallback(WandbCallback):
    def __init__(self, trainer: Trainer, log_args: Dict[str, Any] = None):
        super().__init__()
        self.trainer = trainer
        self.log_args = log_args

    def calculate_matrix_symmetry_stats(
            self,
            key_matrix: torch.Tensor,
            query_matrix: torch.Tensor,
            num_heads: int,
            head_dim: int,
            embed_dim: int,
            weight_type: str = 'WqWk'
    ) -> Dict[str, float]:
        """
        Calculate the symmetry statistics of the attention matrices.
        :param key_matrix: The key matrix weights.
        :param query_matrix: The query matrix weights.
        :param num_heads: The number of attention heads.
        :param head_dim: The dimension of each attention head.
        :param embed_dim: The embedding dimension.
        :param weight_type: Type of the weight (will be added to key of returned dict)
        :return: A dictionary of symmetry statistics.
        """
        M = get_key_query_matrix(key_matrix, query_matrix, num_heads, head_dim, embed_dim)
        norms = torch.norm(M, p='fro', dim=(1, 2))
        all_sym_scores, _ = to_symmetric_asymmetric_fast(key_matrix, query_matrix, num_heads, head_dim, embed_dim)
        # stats = {f'Head {head_index} Symmetry': sym_score for head_index, sym_score in enumerate(all_sym_scores)}
        stats = {f'Head {head_index} Symmetry {weight_type}': all_sym_scores[head_index] for head_index in
                 range(all_sym_scores.shape[0])}
        # stats.update({f'Head {head_index} L2-Norm': norm for head_index, norm in enumerate(norms)})
        stats.update(
            {f'Head {head_index} L2-Norm {weight_type}': norms[head_index] for head_index in range(norms.shape[0])})

        return stats

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model=model, **kwargs)
        if self.log_args is not None:
            self._wandb.config.update(self.log_args, allow_val_change=True)

    def on_log(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        logs = {}
        count = 0
        prev_Wv = []
        for module in self.trainer.model.modules():
            if (isinstance(module, ElectraSelfAttention)
                    or isinstance(module, BertSelfAttention)
                    or isinstance(module, RobertaSelfAttention)
                    or isinstance(module, BertSelfAttentionSymmetricInit)):
                num_heads = module.num_attention_heads
                head_dim = module.all_head_size // module.num_attention_heads
                embed_dim = module.all_head_size
                symmetry_stats = self.calculate_matrix_symmetry_stats(module.key, module.query, num_heads, head_dim,
                                                                      embed_dim)
                prev_Wv.append((module.value, num_heads, head_dim, embed_dim))

            elif isinstance(module, BertSelfOutput) or isinstance(module, ElectraSelfOutput):
                assert len(prev_Wv) == 1
                count -= 1
                Wv, num_heads, head_dim, embed_dim = prev_Wv.pop()
                Wo = module.dense
                symmetry_stats.update(
                    self.calculate_matrix_symmetry_stats(Wv, Wo, num_heads, head_dim, embed_dim, weight_type='WvWo'))

            else:
                continue

            symmetry_stats = {f"Layer {count}/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in
                              symmetry_stats.items() if 'Layer' not in k}
            logs = {**logs, **symmetry_stats}
            count += 1

        for wtype in ['WqWk', 'WvWo']:
            symmetries = np.array([v for k, v in logs.items() if f'Symmetry {wtype}' in k])
            logs[f'Overall Symmetry/Mean Symmetry {wtype}'] = np.mean(symmetries)
            logs[f'Overall Symmetry/Min Symmetry {wtype}'] = np.min(symmetries)
            logs[f'Overall Symmetry/Max Symmetry {wtype}'] = np.max(symmetries)
            logs[f'Overall Symmetry/Median Symmetry {wtype}'] = np.median(symmetries)
            logs[f'Overall Symmetry/Variance Symmetry {wtype}'] = np.var(symmetries)

        if hasattr(self.trainer.model, 'loss_stats'):
            logs.update(self.trainer.model.loss_stats)

        self._wandb.log(logs)
