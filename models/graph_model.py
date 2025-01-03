import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from model_utils import MLP, EdgeGINConv, EdgeGCNConv


class GNN(nn.Module):
    """
    Composition of stacked conv blocks.
    """
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(EdgeGINConv(MLP(input_dim=emb_dim, hidden_dims=2*emb_dim, output_dim=emb_dim, batch_norm=True), BondEncoder(emb_dim=emb_dim), train_eps=True))
            elif gnn_type == "gcn":
                self.gnns.append(EdgeGCNConv(MLP(input_dim=emb_dim, hidden_dims=None, output_dim=emb_dim), BondEncoder(emb_dim=emb_dim), torch.empty(emb_dim)))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(self, num_layer, emb_dim, num_tasks, JK, graph_pooling, molecule_node_model=None):
        super(GNN_graphpred, self).__init__()

        if num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_node_model = molecule_node_model
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.JK = JK

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        print("Loading GNN_graphpred pretrain weight from {} ...".format(model_file))
        state_dict = torch.load(model_file)
        self.molecule_node_model.load_state_dict(state_dict)
        return

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_node_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)  # 64,300
        output = self.graph_pred_linear(graph_representation) # 64,1
        return graph_representation, output