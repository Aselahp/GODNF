#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def segment_sparsemax(scores, index, num_nodes):
    order = torch.argsort(scores, descending=True)
    perm = order[torch.argsort(index[order], stable=True)]
    z, seg = scores[perm], index[perm]

    counts = torch.bincount(seg, minlength=num_nodes)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    rank = torch.arange(seg.numel(), device=scores.device) - ptr[seg]

    gc = z.cumsum(0)
    csum = gc - (gc - z)[ptr[seg]]                      # within-segment cumsum
    support = (1.0 + (rank + 1).to(z.dtype) * z) > csum
    k = scatter_add(support.to(z.dtype), seg, dim=0, dim_size=num_nodes).clamp(min=1)

    tau_idx = (ptr[:-1] + k.long() - 1).clamp(0, z.numel() - 1)
    tau = (csum[tau_idx] - 1.0) / k
    out = torch.clamp(z - tau[seg], min=0.0)

    w = torch.empty_like(out)
    w[perm] = out
    return w


class GODNFLayer(nn.Module):

    def __init__(self, in_features, hidden_features, num_nodes, num_edges,
                 alpha, init_mu, learn_mu=True, t_max=10, use_static_weights=False):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features * 3),
            nn.ReLU(),
            nn.Linear(hidden_features * 3, hidden_features)
        )

        self.alpha = alpha
        self.num_nodes = num_nodes
        self.t_max = t_max
        self.use_static_weights = use_static_weights

        self.node_selection = nn.Parameter(torch.rand(num_nodes))

        self.edge_scores = nn.Parameter(torch.rand(num_edges) * 0.1 + 0.01)

        if not use_static_weights:
            self.register_buffer('delta_cache', torch.zeros(max(t_max - 1, 1), num_edges))
        self._score_leaves = []

        init_raw = torch.log(torch.expm1(torch.tensor(float(init_mu)).clamp(min=1e-4)))
        if learn_mu:
            self.mu_raw = nn.Parameter(init_raw)
        else:
            self.register_buffer('mu_raw', init_raw)

        self.reset_parameters()

    @property
    def mu(self):
        return F.softplus(self.mu_raw)

    def eta(self, t):
        return 1.0 / (1.0 + t)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_laplacian(self, edge_index, num_nodes):
        row, col = edge_index
        deg = scatter_add(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        lap_values = -deg_inv_sqrt[row] * deg_inv_sqrt[col]
        diag = torch.arange(num_nodes, device=edge_index.device)
        lap_indices = torch.cat([edge_index, torch.stack([diag, diag])], dim=1)
        lap_values = torch.cat([lap_values, torch.ones(num_nodes, device=edge_index.device)])
        return lap_indices, lap_values

    def build_score_sequence(self):
        seq = [self.edge_scores]
        self._score_leaves = []
        if not self.use_static_weights:
            prev = self.edge_scores.detach()
            for t in range(1, self.t_max):
                nxt = (prev + self.eta(t - 1) * self.delta_cache[t - 1]).detach()
                if self.training:
                    nxt = nxt.requires_grad_(True)
                    self._score_leaves.append(nxt)
                seq.append(nxt)
                prev = nxt.detach()
        return seq

    @torch.no_grad()
    def cache_gradients(self):
        if self.use_static_weights:
            return
        if self.edge_scores.grad is not None:
            self.delta_cache[0].copy_(self.edge_scores.grad)
        for t, leaf in enumerate(self._score_leaves, start=1):
            if t < self.delta_cache.size(0) and leaf.grad is not None:
                self.delta_cache[t].copy_(leaf.grad)

    def compute_operator_norm_bound_regularization(self, edge_index, w_values, lap_indices, lap_values, S):
        I_minus_S = 1 - S
        m_w = I_minus_S[edge_index[0]] * w_values
        m_l = -self.mu * I_minus_S[lap_indices[0]] * lap_values

        row_sums = torch.zeros(self.num_nodes, device=edge_index.device)
        col_sums = torch.zeros(self.num_nodes, device=edge_index.device)
        row_sums = row_sums.scatter_add(0, edge_index[0], m_w.abs())
        col_sums = col_sums.scatter_add(0, edge_index[1], m_w.abs())
        row_sums = row_sums.scatter_add(0, lap_indices[0], m_l.abs())
        col_sums = col_sums.scatter_add(0, lap_indices[1], m_l.abs())

        bound = torch.sqrt(row_sums.max() * col_sums.max())
        return F.relu(bound - 1.0)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        X_0 = self.mlp(x)
        X_t = X_0

        s_values = torch.sigmoid(self.node_selection)
        lap_indices, lap_values = self.compute_laplacian(edge_index, num_nodes)
        score_seq = self.build_score_sequence()

        reg_loss = X_0.new_zeros(())
        row_w, col_w = edge_index
        row_l, col_l = lap_indices

        for t in range(self.t_max):
            scores = score_seq[0] if self.use_static_weights else score_seq[t]
            w_values = segment_sparsemax(scores, row_w, num_nodes)

            if (not self.use_static_weights) or t == 0:
                reg_loss = reg_loss + self.compute_operator_norm_bound_regularization(
                    edge_index, w_values, lap_indices, lap_values, s_values)

            nbr = scatter_add(w_values.unsqueeze(-1) * X_t[col_w], row_w, dim=0, dim_size=num_nodes)
            lap = scatter_add(lap_values.unsqueeze(-1) * X_t[col_l], row_l, dim=0, dim_size=num_nodes)

            influence = nbr - self.mu * lap
            X_t = self.alpha * X_t + (1 - self.alpha) * (
                s_values.unsqueeze(1) * X_0 + (1 - s_values).unsqueeze(1) * influence
            )

        return X_t, reg_loss


class GODNF(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers,
                 num_nodes, num_edges, adj=0, alpha=0.9, init_mu=0.8, learn_mu=True,
                 use_static_weights=True):
        super().__init__()

        # GNN layer
        self.gnn = GODNFLayer(
            in_features=in_channels,
            hidden_features=hidden_channels,
            num_nodes=num_nodes,
            num_edges=num_edges,
            alpha=alpha,
            init_mu=init_mu,
            learn_mu=learn_mu,
            t_max=num_layers,
            use_static_weights=use_static_weights
        )

        self.dropout = dropout

        self.output_layer1 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        node_embeddings, reg_loss = self.gnn(x, edge_index)

        node_embeddings = F.dropout(node_embeddings, self.dropout, training=self.training)
        out = self.output_layer1(node_embeddings)
        x = torch.sigmoid(out).mean(dim=1, keepdim=True)

        return x, reg_loss