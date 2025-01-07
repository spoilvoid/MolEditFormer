import os
import os.path as osp
import copy
import numpy as np
from typing import Callable, Optional, Union, Any, List
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.graphproppred.mol_encoder import BondEncoder


def cycle_index(num, shift):
    '''
    num, shift: int
    num > shift > 0
    return [shift, shift+1, ..., num-1, 0, 1, ..., shift-1]
    '''
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def pad_array_1d(array_1d, size, value=0):
    """
    Pad an array to a given size.

    Args:
    array_1d (List/np.ndarray): 1D array to pad.
    size (int): Size of the output array.
    value (int): Value to pad with.

    Returns:
    pad_array_1d (np.ndarray): padded 1D array.
    """
    pad_length = size - len(array_1d)
    return np.pad(array_1d, pad_width=(0, pad_length), mode='constant', constant_values=value)


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling of token embeddings.

    Args:
        token_embeddings (torch.Tensor): Token embeddings, 1st dimension indicates max_seq_len. [max_seq_len, batch_size, d_model]
        attention_mask (torch.Tensor): Attention mask, 1 indicates useful while 0 indicates pad token. [max_seq_len, batch_size]

    Output:
        mean_embeddings (torch.Tensor): Mean-pooled embeddings. [batch_size, d_model]
    """
    if token_embeddings.size(0) != attention_mask.size(0) or token_embeddings.size(1) != attention_mask.size(1):
        raise ValueError("The first two dimensions of token_embeddings and attention_mask must be the same.")
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9)
    return sum_embeddings / sum_mask


class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    """
    A multi-layer perceptron with at least 2 layers that allows customization num_layers,  width_layer of each layer, activation, dropout, and normalization.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, batch_norm=False, activation="relu", dropout=0):
        """
        Args:
        input_dim (int): Input's last Dimension.
        hidden_dims (List[int]/Tuple(int)/int/None): Dimensions of the hidden layers
        output_dim (int): Output's Last Dimension.
        batch_norm (bool): Whether to use batch normalization, default is False.
        activation (str): Activation function Name, default is "relu".
        dropout (float): Dropout rate, default is 0.
        """
        super(MLP, self).__init__()

        if hidden_dims == None:
            hidden_dims = []
        elif not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims + [output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.batch_norms:
            for bn in self.batch_norms:
                bn.reset_parameters()

    def forward(self, input):
        """
        Args:
        input (torch.tensor): Input of the multi-layer perceptron, its last Dimension is input_dim.
        
        Returns:
        output (torch.tensor): Output of the multi-layer perceptron, its last Dimension is output_dim, the rest Dimension is the same as input.
        """
        layer_input = input
        
        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    # flatten the hidden tensor to apply batch_norm and then turn back to the original shape
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            # skip connection
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class EdgeGINConv(MessagePassing):
    """
    GINConv Block that could handle Edge features with variable atom encoder and bond encoder
    """
    def __init__(
        self, 
        atom_encoder: Callable, 
        bond_encoder: Optional[Callable] = None, 
        eps: float = 0., 
        train_eps: bool = False, 
        **kwargs,
    ):
        """
        Args:
        atom_encoder (Callable): nn.Module for Atom Feature Encoder.
        bond_encoder (Callable): nn.Module for Bond Feature Encoder.
        eps (float): Initial value of epsilon, default is 0.
        train_eps (bool): Whether to train epsilon, default is False.
        **kwargs: Additional arguments of torch_geometric.nn.conv.MessagePassing.
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.atom_encoder = atom_encoder
        if bond_encoder is not None:
            self.bond_encoder = bond_encoder
        else:
            self.bond_encoder = None
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.atom_encoder.reset_parameters()
        if self.bond_encoder is not None:
            self.bond_encoder.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.atom_encoder(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.bond_encoder is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not match. Consider resetting the AtomEncoder and BondEncoder")

        if self.bond_encoder is not None:
            edge_attr = self.bond_encoder(edge_attr)

        return (x_j + edge_attr).relu()


# 实现方式与官方的GCN有区别，主要是在对propagate后的处理上
class EdgeGCNConv(MessagePassing):
    """
    GCNConv Block that could handle Edge features with variable atom encoder and bond encoder
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        atom_encoder: Callable, 
        bond_encoder: Optional[Callable] = None,
        bias: OptTensor = None, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Args:
        atom_encoder (Callable): nn.Module for Atom Feature Encoder.
        bond_encoder (Callable): nn.Module for Bond Feature Encoder.
        bias (OptTensor): Bias tensor, default is None.
        improved, cached, add_self_loops, normalize are arguments of basic GCNConv.
        **kwargs: Additional arguments of torch_geometric.nn.conv.MessagePassing.
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support adding self-loops to the graph when no on-the-fly normalization is applied")

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.atom_encoder = atom_encoder
        if bond_encoder is not None:
            self.bond_encoder = bond_encoder
        if bias is not None:
            self.bias = Parameter(copy.deepcopy(bias))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.atom_encoder.reset_parameters()
        if self.bond_encoder is not None:
            self.bond_encoder.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple of node features as input while this layer does not support bipartite message passing. Please try other layers such as 'SAGEConv' or 'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # row, col = edge_index
        # deg = degree(row, x.size(0), dtype = x.dtype) + 1
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

        x = self.atom_encoder(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_attr: Tensor) -> Tensor:
        if self.bond_encoder is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not match. Consider resetting the AtomEncoder and BondEncoder")
        
        if self.bond_encoder is not None:
            edge_attr = self.bond_encoder(edge_attr)

        return (x_j + edge_attr).relu() if edge_weight is None else edge_weight.view(-1, 1) * (x_j + edge_attr).relu()


if __name__=="__main__":
    input_dim = 300
    emb_dim = 300
    # EdgeGINConv Usage
    gin_atom_encoder = MLP(input_dim=input_dim, hidden_dims=2*emb_dim,output_dim=emb_dim, batch_norm=True)
    gin_bond_encoder = BondEncoder(emb_dim=emb_dim)
    edge_gin_model = EdgeGINConv(atom_encoder=gin_atom_encoder, bond_encoder=gin_bond_encoder, train_eps=True)
    # EdgeGCNConv Usage
    # gin_atom_encoder = MLP(input_dim=input_dim, hidden_dims=None,output_dim=emb_dim)
    # gin_bond_encoder = BondEncoder(emb_dim=emb_dim)
    # bias = torch.empty(emb_dim)
    # edge_gcn_model = EdgeGCNConv(atom_encoder=gin_atom_encoder, bond_encoder=gin_bond_encoder, bias=bias)
    pass
