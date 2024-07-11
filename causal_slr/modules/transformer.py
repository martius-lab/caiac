import itertools
from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from causal_slr.components.base_model import BaseModel
from causal_slr.modules import FactorizedMLP


class Transformer(FactorizedMLP, BaseModel):
    def __init__(self,
                 inp_dim: 'OrderedDict[str, Any]',
                 outp_dim: 'OrderedDict[str, Any]',
                 embedding_dim,
                 n_layers,
                 n_heads=1,
                 fc_dim=64,
                 outp_layer=nn.Linear,
                 dropout_probs=None,
                 dropout=0.0,
                 bn_first=False,
                 logger=None,
                 loss_fn=None,
                 device=None,
                 use_state_diff_as_target=False,
                 input_factorizer=None,
                 **kwargs):
        BaseModel.__init__(self, logger)
        self.__device = None
        # Convert first to dict so that the order of the keys is preserved.
        self.outp_groups = {
            name: "" for name in inp_dim.keys() if name in outp_dim.keys()}.keys()
        self.only_inp_groups = {name: "" for name in inp_dim.keys()
                                if name not in outp_dim.keys()}.keys()
        self.inp_dims = OrderedDict(inp_dim)
        self.outp_dims = OrderedDict(outp_dim)
        self.loss_fn = loss_fn
        self.use_state_diff_as_target = use_state_diff_as_target
        # not used in the class, but used for computing CODA in mask_scorers.py
        self.input_factorizer = input_factorizer
        # Projection from each input factor to embedding space
        self.inp_projs = nn.ModuleDict()
        for name in itertools.chain(self.outp_groups, self.only_inp_groups):
            shape = inp_dim[name]
            bn = None
            if bn_first:
                bn = nn.BatchNorm1d(shape[0], momentum=0.1, affine=False)

            if bn is None:
                inp_proj = nn.Linear(shape[0], embedding_dim)
            else:
                inp_proj = nn.Sequential(bn,
                                         nn.Linear(shape[0], embedding_dim))
            self.inp_projs[name] = inp_proj

        # Projection from embedding space to each output factor dimensionality
        self.outp_projs = nn.ModuleDict()
        for name in self.outp_groups:
            shape = outp_dim[name]
            self.outp_projs[name] = outp_layer(embedding_dim, shape[0])

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(embedding_dim,
                                            nhead=n_heads,
                                            dim_feedforward=fc_dim,
                                            dropout=dropout)
            self.layers.append(layer)

        if dropout_probs is None:
            dropout_probs = {}

        probs = [[dropout_probs.get(name, 0) for name in self.inp_projs]]
        self.register_buffer('dropout_probs',
                             torch.tensor(probs, dtype=torch.float))

        if device is not None:
            self.device = device

    def load_state_dict(self, state_dict, strict=True):
        if state_dict['dropout_probs'].shape != self.dropout_probs.shape:
            state_dict['dropout_probs'] = self.dropout_probs

        cur_state_dict = self.state_dict()
        missing_inp_keys = {name for name in state_dict
                            if (name.startswith('inp_projs')
                                and name not in cur_state_dict)}
        for key in missing_inp_keys:
            del state_dict[key]

        super().load_state_dict(state_dict, strict=strict)

    def get_input_index(self, key):
        """Index of group named by key in input"""
        return list(self.inp_projs).index(key)

    def get_output_index(self, key):
        """Index of group named by key in output"""
        return list(self.outp_projs).index(key)

    def get_mask(self, x):
        x = self.normalize_(x.copy(), self.norm_params_input)
        # Project each input factor to embedding space and stack them such as embeddings = [num_input_factors x batch_size x embedding_dim] where num_input_factors
        embeddings = [proj(x[name]) for name, proj in self.inp_projs.items()]
        embeddings = torch.stack(embeddings, dim=0)

        joint_mask = None
        atten_masks = []
        for layer in self.layers:
            # mask: Batch_size x Num_outp_factors x Num_input_factors. Out dimensionality is the same as input
            # embeddings: Num_input_factors x Batch_size x Embedding_dim
            embeddings, mask = layer(embeddings,
                                     return_attention_weights=True)
            if joint_mask is None:
                joint_mask = mask.transpose(1, 2)
            else:
                joint_mask = joint_mask.bmm(mask.transpose(1, 2))
            atten_masks.append(mask.transpose(1, 2))
        # Returned mask is encoded as Batch_size x Num_Inp x Num_Out
        # select from the columns only the ones that correspond to the outputs we predict
        return joint_mask[:, :, :len(self.outp_projs)], atten_masks

    def forward_(self, x):
        embeddings = [proj(x[name]) for name, proj in self.inp_projs.items()]
        embeddings = torch.stack(embeddings, dim=0)

        if self.training and self.dropout_probs is not None:
            bs = embeddings.shape[1]
            probs = self.dropout_probs.expand(bs, -1)
            mask = torch.bernoulli(probs)
        else:
            mask = None

        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=mask)
        outp = OrderedDict()
        for idx, (name, proj) in enumerate(self.outp_projs.items()):
            # `embeddings` encoded as Target_dim x Batch_size x Embedding_dim
            outp[name] = proj(embeddings[idx])

        # Dict with keys out_dims and values of shape (batch_size, out_dim)
        return outp

    def forward(self, x):
        x_ = self.normalize_(x.copy(), self.norm_params_input)

        outp = self.forward_(x_)
        outp = self.denormalize_(outp, self.norm_params_target)
        with torch.no_grad():
            if self.use_state_diff_as_target:
                for key in self.outp_dims.keys():
                    # Only add the offset to the 'mean' not 'var'
                    try:
                        outp[0][key] = outp[0][key] + x[key]
                    except RuntimeError:
                        print(
                            f'Dimension mismatch between input and output for key {key}. Input[{key}] has shape {x[key].shape} vs output key {key} has shape {outp[0][key].shape}   Assuming only predict first k elements of input')
                        outp[0][key] = outp[0][key] + \
                            x[key][:, :outp[0][key].shape[1]]
        return outp

    def compute_loss(self, input, target):
        inp = self.normalize_(input, self.norm_params_input)
        target = self.normalize_(target, self.norm_params_target)

        output = self.forward_(inp)

        assert output.keys() == target.keys(), "Output and target keys must match"
        # Iterate over same keys for target and output such that concatenation gets same order.
        target_ = torch.cat([target[key]
                            for key in target.keys()], axis=-1)
        output_ = torch.cat([output[key]
                            for key in target.keys()], axis=-1)

        loss = self.loss_fn(output_, target_).mean()

        return loss

    def denormalize_(self, input_, norm_params):
        mean, var = {}, None
        for key in self.outp_dims:
            mean[key] = input_[key] * norm_params['std'][key] + \
                norm_params['mean'][key]
        input = (mean, var)
        return input


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Same as `nn.TransformerEncoderLayer`, but returns attention weights"""

    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)

    # See line 478 and then 586 in transformer.py from nn.modules. Exactly the same but reorganized + return mask
    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                return_attention_weights=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
            src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` (default!). 
            `S` is the source sequence length (num of entities), 'N' is the batch size, `E` is the embedding dimension.
        """
        # input to self_attention is (query, key, value, ...)

        """ "
        Outputs:
        - src: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: If ``average_attn_weights=True`` (default), returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length.  S=L in our case.
        """
        src2, mask = self.self_attn(src, src, src,
                                    attn_mask=src_mask,
                                    need_weights=return_attention_weights,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attention_weights:
            return src, mask
        else:
            return src
