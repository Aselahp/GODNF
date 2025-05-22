
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import time
import random
import pickle
import pandas as pd
import scipy.sparse as sp
import os
from torch.optim import Adam
from est_models import GODNF


# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

# Define available datasets and diffusion models
datasets = ['jazz', 'netscience', 'cora_ml', 'power_grid']
diffusion_models = ['IC', 'LT', 'SIS']

seed_rate = 10
mode = "normal"
n = 3 # number of layers

# Create a results dataframe to store all results
all_results = pd.DataFrame(columns=['Dataset', 'Diffusion_Model', 'Mean_Error', 'Std_Dev', 'Val_Errors'])

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def estimation_loss(y, y_hat):
    forward_loss = F.mse_loss(y_hat.squeeze(), y, reduction='sum')
    return forward_loss

# Main loop for different datasets and diffusion models
for dataset_name in datasets:
    for diffusion_model in diffusion_models:
        print(f"\n{'='*80}")
        print(f"Running experiment with Dataset: {dataset_name}, Diffusion Model: {diffusion_model}")
        print(f"{'='*80}\n")
        
        try:
            # Check if the file exists before trying to load it
            file_path = f'data/{dataset_name}_mean_{diffusion_model}{10*seed_rate}.SG'
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping this combination.")
                continue
                
            # Load the graph data
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)

            adj, dataset = graph['adj'], graph['inverse_pairs']

            # Process adjacency matrix
            adj_m = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj_m = normalize_adj(adj_m + sp.eye(adj_m.shape[0]))
            adj_m = torch.Tensor(adj_m.toarray()).to_sparse()

            # Generate feature matrix
            adjacency_matrix = torch.sparse_coo_tensor(
                torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long),
                torch.tensor(adj.tocoo().data, dtype=torch.float32),
                torch.Size(adj.tocoo().shape)
            )

            two_degree = torch.sparse.mm(adjacency_matrix, adjacency_matrix)
            three_degree = torch.sparse.mm(two_degree, adjacency_matrix)
            degree = (torch.sparse.sum(adjacency_matrix, dim=1)).to_dense()
            unique_degrees = torch.unique(degree)

            one_hot_encoder = {deg.item(): i for i, deg in enumerate(unique_degrees)}
            num_unique_degrees = len(unique_degrees)
            num_nodes = adjacency_matrix.size(0)
            feature_matrix = torch.zeros((num_nodes, num_unique_degrees))

            for i, deg in enumerate(degree):
                one_hot_index = one_hot_encoder[deg.item()]
                feature_matrix[i, one_hot_index] = 1.0

            adj = torch.Tensor(adj.toarray()).to_sparse()
            adj = adj.to(device)
            feature_matrix = feature_matrix.to(device)

            edge_index = adj.coalesce().indices()

            # Set batch size based on dataset
            batch_size = 10

            # Cross-validation setup
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            val_error = []

            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                print(f'Fold {fold + 1}')
                
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
                
                model = GODNF(
                    in_channels=feature_matrix.shape[1], 
                    hidden_channels=32, 
                    out_channels=1, 
                    dropout=0.5, 
                    adj=adj, 
                    num_layers=n, 
                    num_nodes=feature_matrix.shape[0]
                ).to(device)
    
                optimizer = Adam([{'params': model.parameters()}], lr=0.01, weight_decay=5e-4)
                
                # Training loop
                for epoch in range(200):
                    begin = time.time()
                    total_overall = 0
                    model.train()

                    for batch_idx, data_pair in enumerate(train_loader):
                        optimizer.zero_grad()
                        
                        x = data_pair[:, :, 0].float().to(device)
                        y = data_pair[:, :, 1].float().to(device)
                        
                        loss = 0
                        for i, x_i in enumerate(x):
                            y_i = y[i]
                            
                            x_hat = feature_matrix
                            y_hat, curv_loss = model(x_hat, edge_index)
                            total = estimation_loss(y_i, y_hat) + curv_loss
                                        
                            loss += total

                        total_overall += loss.item()
                        loss = loss/x.size(0)
                        loss.backward()
                        optimizer.step()
                        
                    end = time.time()
                    print("Epoch: {}".format(epoch+1), 
                        "\tTotal: {:.4f}".format(total_overall / len(train_subset)),
                        "\tTime: {:.4f}".format(end - begin)
                        )
                
                # Validation
                val_mae = 0
                model.eval()
                with torch.no_grad():
                    for batch_idx, data_pair in enumerate(val_loader):
                        x = data_pair[:, :, 0].float().to(device)
                        y = data_pair[:, :, 1].float().to(device)

                        x_hat = feature_matrix
                        y_hat, curv_loss = model(x_hat, edge_index)
                        val_mae += np.abs(y_hat.squeeze() - y[0]).sum()/x[0].shape[0]
                
                val_mae /= len(val_loader)
                val_error.append(val_mae)
                print('Validation Loss: ', val_mae)
            
            # Calculate final results for this combination
            mean = np.mean(val_error)
            std_dev = np.std(val_error, ddof=1)  

            print(f"\nResults for {dataset_name} with {diffusion_model}:")
            print(f"Mean: {mean}")
            print(f"Standard Deviation: {std_dev}")
            
            new_row = pd.DataFrame({
                'Dataset': [dataset_name],
                'Diffusion_Model': [diffusion_model],
                'Mean_Error': [mean],
                'Std_Dev': [std_dev],
                'Val_Errors': [val_error]
            })
            all_results = pd.concat([all_results, new_row], ignore_index=True)
            
            # Clear GPU memory between runs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {dataset_name} with {diffusion_model}: {str(e)}")
            continue

# Save all results to CSV
all_results.to_csv('GODNF_dynamic_results.csv', index=False)
