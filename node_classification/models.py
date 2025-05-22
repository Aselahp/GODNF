import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add


    
class GODNFLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_nodes, alpha, init_mu, learn_mu=True, t_max=10, use_static_weights=False):
        super(GODNFLayer, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features*3),
            nn.ReLU(),
            nn.Linear(hidden_features*3, hidden_features)
        )
         
        self.alpha = alpha
        self.register_parameter('node_selection', nn.Parameter(torch.rand(num_nodes)))
        self.num_nodes = num_nodes
        
        if learn_mu:
            self.register_parameter('mu', nn.Parameter(torch.tensor(init_mu)))
        else:
            self.register_buffer('mu', torch.tensor(init_mu))
        
        self.t_max = t_max
        self.eta = lambda t: 1.0 / (1.0 + t)
        self.use_static_weights = use_static_weights
        
        if use_static_weights:
            self.static_weight = None  
        else:
            self.dynamic_weights = nn.ParameterDict({})
        
        self.reset_parameters()
    
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
        
        lap_indices = edge_index.clone()
        lap_values = -deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        diag_indices = torch.stack([torch.arange(num_nodes, device=edge_index.device), 
                                    torch.arange(num_nodes, device=edge_index.device)])
        diag_values = torch.ones(num_nodes, device=edge_index.device)
        
        lap_indices = torch.cat([lap_indices, diag_indices], dim=1)
        lap_values = torch.cat([lap_values, diag_values])
        
        return lap_indices, lap_values
    
    def get_weight_values(self, edge_index, t):
        num_edges = edge_index.size(1)
        if self.use_static_weights:
            if self.static_weight is None:
                self.static_weight = nn.Parameter(
                    torch.rand(num_edges, device=edge_index.device) * 0.1 + 0.01
                )
                self.register_parameter('static_weight', self.static_weight)
            raw_weights = self.static_weight
        
        else:
            edge_key = f't{t}'
            if edge_key not in self.dynamic_weights:
                self.dynamic_weights[edge_key] = nn.Parameter(
                    torch.rand(num_edges, device=edge_index.device) * 0.1 + 0.01
                )
            
            raw_weights = self.dynamic_weights[edge_key]
        
        positive_weights = F.relu(raw_weights) + 1e-5
        src, dst = edge_index
        row_sum = scatter_add(positive_weights, src, dim=0, dim_size=self.num_nodes)
        row_sum_inv = 1.0 / (row_sum + 1e-8)
        normalized_weights = positive_weights * row_sum_inv[src]
        
        return edge_index, normalized_weights
    
    def update_weights(self, t, edge_index, loss):
        if self.use_static_weights:
            return       
        edge_key = f't{t}'
        next_key = f't{t+1}'     
        if next_key in self.dynamic_weights:
            return
        
        if edge_key not in self.dynamic_weights:
            _, _ = self.get_weight_values(edge_index, t)
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            eta_t = self.eta(t)
            if self.dynamic_weights[edge_key].grad is not None:
                delta_weights = self.dynamic_weights[edge_key].grad
                new_weights = self.dynamic_weights[edge_key].detach() + eta_t * delta_weights
                new_weights = F.relu(new_weights) + 1e-5
                src, _ = edge_index
                row_sum = scatter_add(new_weights, src, dim=0, dim_size=self.num_nodes)
                row_sum_inv = 1.0 / (row_sum + 1e-8)
                normalized_weights = new_weights * row_sum_inv[src]
                
                self.dynamic_weights[next_key] = nn.Parameter(normalized_weights)
                self.dynamic_weights[edge_key].grad.zero_()
    
    def compute_operator_norm_bound_regularization(self, edge_index, num_nodes, S, t):
        w_indices, w_values = self.get_weight_values(edge_index, t)
        l_indices, l_values = self.compute_laplacian(edge_index, num_nodes)
        I_minus_S = 1 - S
        factor = 1 
        m_w_values = factor * I_minus_S[w_indices[0]] * w_values
        m_l_values = -factor * self.mu * I_minus_S[l_indices[0]] * l_values
        row_sums = torch.zeros(num_nodes, device=edge_index.device)
        col_sums = torch.zeros(num_nodes, device=edge_index.device)
        row_sums.scatter_add_(0, w_indices[0], torch.abs(m_w_values))
        col_sums.scatter_add_(0, w_indices[1], torch.abs(m_w_values))
        row_sums.scatter_add_(0, l_indices[0], torch.abs(m_l_values))
        col_sums.scatter_add_(0, l_indices[1], torch.abs(m_l_values))
        max_row_sum = row_sums.max()
        max_col_sum = col_sums.max()
        op_norm_bound = torch.sqrt(max_row_sum * max_col_sum)
        reg_loss = F.relu(op_norm_bound - 1.0 + 1e-10) 
        return reg_loss
    
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device
        X_0 = self.mlp(x)
        X_0 = F.dropout(X_0, 0.0, training=self.training)
        X_t = X_0.clone()

        # stubbernness values
        s_values = torch.sigmoid(self.node_selection)
        
        # Compute Laplacian
        lap_indices, lap_values = self.compute_laplacian(edge_index, num_nodes)
        
        reg_loss = 0
        
        if self.use_static_weights:
            w_indices, w_values = self.get_weight_values(edge_index, 0)
            reg_loss = self.compute_operator_norm_bound_regularization(edge_index, num_nodes, s_values, 0)
        else:
            w_indices, w_values = self.get_weight_values(edge_index, 0)
        
        # Main loop
        for t in range(self.t_max):
            if not self.use_static_weights and t > 0:
                w_indices, w_values = self.get_weight_values(edge_index, t)
            
            s_x0 = s_values.unsqueeze(1) * X_0

            row, col = w_indices
            weighted_neighbor_features = scatter_add(
                w_values.unsqueeze(-1) * X_t[col], 
                row, 
                dim=0, 
                dim_size=num_nodes
            )

            row, col = lap_indices
            laplacian_features = scatter_add(
                lap_values.unsqueeze(-1) * X_t[col],
                row,
                dim=0,
                dim_size=num_nodes
            )
            
            neighbor_influence =  weighted_neighbor_features - self.mu * laplacian_features
            one_minus_s = (1 - s_values).unsqueeze(1)
            
            X_t_plus_1 = self.alpha * X_t + (1 - self.alpha) * (
                s_x0 + one_minus_s * neighbor_influence
            )
            
            #regularization term computation
            if not self.use_static_weights:
                reg_loss_t = self.compute_operator_norm_bound_regularization(edge_index, num_nodes, s_values, t)
                reg_loss = reg_loss + reg_loss_t
            
            X_t = F.relu(X_t_plus_1)
        
        return X_t, reg_loss



class GODNF(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 num_nodes, alpha=0.3, init_mu=0.8, learn_mu=True, t_max=10, use_static_weights=False):
        super(GODNF, self).__init__()
        
        # GNN layer
        self.gnn = GODNFLayer(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_nodes=num_nodes,
            alpha=alpha,
            init_mu=init_mu,
            learn_mu=learn_mu,
            t_max=t_max,
            use_static_weights=use_static_weights
        )
        
        self.dropout = dropout

        self.output_layer1 = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        node_embeddings, reg_loss = self.gnn(x, edge_index)
        
        node_embeddings = F.dropout(node_embeddings, self.dropout, training=self.training)
        out = self.output_layer1(node_embeddings)
        
        return F.log_softmax(out, dim=-1), F.sigmoid(reg_loss)


