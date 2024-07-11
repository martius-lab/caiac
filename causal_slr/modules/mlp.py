import torch
from torch import nn
import collections
from causal_slr.utils import conf_utils
from causal_slr.components.base_model import BaseModel
from causal_slr.utils.general_utils import rec_map_dict


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    else:
        raise NotImplementedError


def init_module(m, w_init, b_init):
    if hasattr(m, 'initialized'):
        return
    if (hasattr(m, 'weight') and not hasattr(m, 'weight_initialized')
            and m.weight is not None and w_init is not None):
        w_init(m.weight)
    if (hasattr(m, 'bias') and not hasattr(m, 'bias_initialized')
            and m.bias is not None and b_init is not None):
        b_init(m.bias)


class MLP(BaseModel):
    def __init__(self,
                 input_size: int = 1,
                 output_size: int = 1,
                 hidden_dims: list = [],
                 hidden_activation=nn.ReLU,
                 outp_layer='linear',
                 outp_layer_params=dict(),
                 outp_activation=nn.Identity,
                 weight_init=None,
                 bias_init=None,
                 bn_first=True,
                 use_spectral_norm=False,
                 logger=None,
                 **kwargs):
        BaseModel.__init__(self, logger)
        self.__device = None
        self.w_init = conf_utils.get_init_weights(weight_init)
        self.b_init = conf_utils.get_init_weights(bias_init)

        layers = []
        self.input_bn = None
        if bn_first:  # True
            bn = nn.BatchNorm1d(input_size, momentum=0.1, affine=False)
            self.input_bn = bn
        current_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim,
                                    hidden_dim))
            if use_spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            layers.append(hidden_activation())

            current_dim = hidden_dim

        # Output layer
        outp_layer_params['in_features'] = current_dim
        outp_layer_params['out_features'] = output_size
        outp_layer = conf_utils.get_layer(outp_layer, outp_layer_params)
        layers.append(outp_layer)
        if outp_activation is not None:
            layers.append(outp_activation())

        self.layers = nn.Sequential(*layers)
        self.init()

    def init(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.initialized = True
            if isinstance(module, nn.LayerNorm):
                module.initialized = True
        self.layers.apply(lambda m: init_module(m, self.w_init, self.b_init))

    def forward(self, inp):
        if self.input_bn is not None:
            inp = self.input_bn(inp)

        out = self.layers(inp)

        return out

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.to(device)
        self.__device = device


class FactorizedMLP(MLP):
    """MLP taking factorized inputs"""

    def __init__(self, inp_dim, outp_dim, device, loss_fn=None, use_state_diff_as_target=False, input_factorizer=None, **kwargs,):
        # {'state': dim_state, 'action': dim_action}
        self.inp_dims = collections.OrderedDict(inp_dim)
        self.outp_dims = collections.OrderedDict(outp_dim)

        inp_dim = sum(shape[0] for shape in inp_dim.values())
        outp_dim = sum(shape[0] for shape in outp_dim.values())
        super().__init__(inp_dim, outp_dim, **kwargs)
        if device is not None:
            self.device = device
        self.use_state_diff_as_target = use_state_diff_as_target
        self.loss_fn = loss_fn
        self.output_factorizer = self.get_output_factorizer
        self.input_factorizer = input_factorizer

    def get_output_factorizer(self, x):
        i = 0
        d = collections.OrderedDict()
        for k, dim in self.outp_dims.items():
            d[k] = x[..., i:i + dim[0]]
            i += dim[0]
        return d

    def forward(self, input):
        # important the copy! otherwise it modifies input too!
        inp = self.normalize_(input.copy(), self.norm_params_input)
        inp = torch.cat([inp[key] for key in self.inp_dims], axis=-1)

        output = super().forward(inp)
        output = (self.output_factorizer(
            output[0]), self.output_factorizer(output[1]))
        output = self.denormalize_(output, self.norm_params_target)
        with torch.no_grad():
            if self.use_state_diff_as_target:
                for key in self.outp_dims.keys():
                    # Only add the offset to the 'mean' not 'var'
                    shape = output[0][key].shape[-1]
                    output[0][key] = output[0][key] + input[key][:, 0:shape]

        return output

    def compute_loss(self, input, target):
        inp = self.normalize_(input, self.norm_params_input)
        target = self.normalize_(target, self.norm_params_target)
        inp = torch.cat([inp[key] for key in self.inp_dims], axis=-1)
        output = super().forward(inp)
        target = torch.cat([target[key]
                           for key in self.outp_dims.keys()], axis=-1)

        loss = self.loss_fn(output, target).mean()

        return loss

    def normalize_(self, input_, norm_params):
        if isinstance(norm_params['mean'], torch.Tensor) and norm_params['mean'] == 0.:
            print('QUICK FIX for old models without per object mean and std!')
            self.norm_params_input['mean'] = {}
            self.norm_params_input['std'] = {}
            for k in input_.keys():
                self.norm_params_input['mean'][k] = torch.zeros(
                    self.inp_dims[k], device=self.device)
                self.norm_params_input['std'][k] = torch.ones(
                    self.inp_dims[k], device=self.device)
        for key in self.inp_dims:
            if key in norm_params['mean'].keys():
                input_[key] = (input_[key]-norm_params['mean']
                               [key])/norm_params['std'][key]
        return input_

    def denormalize_(self, input_, norm_params):
        if isinstance(norm_params['mean'], torch.Tensor) and norm_params['mean'] == 0.:
            print('QUICK FIX for old models without per object mean and std!')
            self.norm_params_target['mean'] = {}
            self.norm_params_target['std'] = {}
            for k in input_[0].keys():
                self.norm_params_target['mean'][k] = torch.zeros(
                    self.outp_dims[k], device=self.device)
                self.norm_params_target['std'][k] = torch.ones(
                    self.outp_dims[k], device=self.device)
        mean, var = {}, {}
        for key in self.outp_dims:
            mean[key] = input_[0][key] * \
                norm_params['std'][key] + norm_params['mean'][key]
            var[key] = input_[1][key] * (norm_params['std'][key])**2
        input = (mean, var)
        return input

    def norms_to_device(self, device):
        self.norm_params_input['mean'] = rec_map_dict(lambda x: torch.Tensor(
            x).to(device), self.norm_params_input['mean'])
        self.norm_params_input['std'] = rec_map_dict(lambda x: torch.Tensor(
            x).to(device), self.norm_params_input['std'])
        self.norm_params_target['mean'] = rec_map_dict(lambda x: torch.Tensor(
            x).to(device), self.norm_params_target['mean'])
        self.norm_params_target['std'] = rec_map_dict(lambda x: torch.Tensor(
            x).to(device), self.norm_params_target['std'])
