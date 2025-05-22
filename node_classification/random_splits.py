from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import GODNF
from dataset_utils import DataLoader

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--train_rate', type=float, default=0.6)
parser.add_argument('--val_rate', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='citeseer', help='Dataset name.')
parser.add_argument('--use_static_weights', type=bool, default=True, help='Static/dynamic weights.')


args = parser.parse_args("")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

dname = args.dataset
dataset = DataLoader(dname)
data = dataset[0]

train_rate = args.train_rate
val_rate = args.val_rate
percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
val_lb = int(round(val_rate*len(data.y)))

permute_masks = random_planetoid_splits

def create_random_split(data, run_seed):
    np.random.seed(args.seed + run_seed)
    torch.manual_seed(args.seed + run_seed)
    
    
    data_split = permute_masks(data, dataset.num_classes, percls_trn, val_lb)
    
    # Get all required data
    A, X, labels, idx_train, idx_val, idx_test = load_citation_data(data_split)
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(labels)
       
    return data_split, features, labels, A, idx_train, idx_val, idx_test


def copy_data(data):
    """Create a deep copy of the dataset object"""
    import copy
    return copy.deepcopy(data)


def train(epoch, model, optimizer, features, adj, labels, idx_train, count):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, reg_loss = model(features, adj)
    loss_train = F.nll_loss(output[idx_train].squeeze(-1), labels[idx_train]) + reg_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train.item(), acc_train.item()


def test(model, features, adj, labels, data_split):
    model.eval()
    logits, curv_loss = model(features, adj)
    accs, losses, preds = [], [], []
    for _, mask in data_split('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = accuracy(logits[mask], labels[mask])
        loss = F.nll_loss(logits[mask].squeeze(-1), labels[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses


# Main experiment loop
Results0 = []
run_stats = []  # Store detailed stats for each run

for i in range(10):
    print(f"\n{'='*50}\nRun {i+1}/10\n{'='*50}")
    
    data, features, labels, adj, idx_train, idx_val, idx_test = create_random_split(data, i)
    model = GODNF(in_features=features.shape[1],
                    hidden_features=args.hidden,
                    out_features=labels.max()+1, #args.hidden,  # Using hidden dim for the embedding
                    num_nodes=features.shape[0],
                    dropout=args.dropout,
                    t_max=2,
                    use_static_weights = args.use_static_weights
                    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    run_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_epoch = 0
    
    # Train the model
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train(epoch, model, optimizer, features, adj, labels, idx_train, epoch)
        run_history['train_loss'].append(train_loss)
        run_history['train_acc'].append(train_acc)
        
        # Evaluate
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, features, adj, labels, data)
        run_history['val_loss'].append(val_loss)
        run_history['val_acc'].append(val_acc)
        
        # Track validation metrics for early stopping
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Check for best validation performance
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
        
        # Early stopping check
        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Record results for this run
    Results0.append([test_acc, best_val_acc])
    run_stats.append({
        'run': i,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc.item(),
        'best_val_loss': best_val_loss.item(),
        'test_acc': test_acc.item(),
        'history': run_history
    })
    
    print(f"Run {i+1} results: Test accuracy = {test_acc:.4f}, Best validation accuracy = {best_val_acc:.4f}")

# Compute statistics over all runs
test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100

print(f"\n{'='*50}\nFinal Results\n{'='*50}")
print(f'Test accuracy: {test_acc_mean:.2f}% ± {test_acc_std:.2f}%')
print(f'Validation accuracy: {val_acc_mean:.2f}%')

# Print individual run results for comparison
print("\nIndividual run results:")
for i, result in enumerate(Results0):
    print(f"Run {i+1}: Test acc = {result[0]*100:.2f}%, Val acc = {result[1]*100:.2f}%")