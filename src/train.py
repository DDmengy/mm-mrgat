#! /usr/bin/env python
# coding: utf-8

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ================= 1. Hyper Parameters =================
parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=80)
parser.add_argument("--n_inp", type=int, default=64)
parser.add_argument("--n_hid", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--mask_ratio", type=float, default=0.3)
parser.add_argument("--fivefold", action='store_true')
parser.add_argument("--graph_path", type=Path, default=DATA_DIR / "heterogeneous_graphs" / "heterographs.bin")
parser.add_argument("--circ_seq_path", type=Path, default=DATA_DIR / "circ_seq_embedding.npy")
parser.add_argument("--dis_sem_path", type=Path, default=DATA_DIR / "dis_sem_embedding.npy")
args = parser.parse_args()

if not args.graph_path.is_file():
    parser.error(f"Graph file is missing: {args.graph_path}. See data/README.md for required inputs.")

import dgl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
from model import MaskedGraphModel


# ================= 2. Path & Logging =================
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# ================= 3. Load Data =================
logging.info("Loading graphs...")
g_list, _ = dgl.load_graphs(str(args.graph_path))
G_tot, G_train, G_test, neg_G_test = g_list

def load_features():
    circ_seq, dis_sem = None, None
    if args.circ_seq_path.is_file():
        logging.info("Loading circRNA sequence features from %s", args.circ_seq_path)
        data = np.load(args.circ_seq_path, allow_pickle=False)
        dim = min(data.shape[1], args.n_inp)
        t = torch.zeros((data.shape[0], args.n_inp))
        t[:, :dim] = torch.tensor(data[:, :dim], dtype=torch.float32)
        circ_seq = t.to(device)
    else:
        logging.warning("CircRNA sequence features are absent: %s", args.circ_seq_path)
    if args.dis_sem_path.is_file():
        logging.info("Loading disease semantic features from %s", args.dis_sem_path)
        data = np.load(args.dis_sem_path, allow_pickle=False)
        dim = min(data.shape[1], args.n_inp)
        t = torch.zeros((data.shape[0], args.n_inp))
        t[:, :dim] = torch.tensor(data[:, :dim], dtype=torch.float32)
        dis_sem = t.to(device)
    else:
        logging.warning("Disease semantic features are absent: %s", args.dis_sem_path)
    return circ_seq, dis_sem

circ_seq, dis_sem = load_features()
node_dict = {nt: i for i, nt in enumerate(G_tot.ntypes)}
edge_dict = {et: i for i, et in enumerate(G_tot.etypes)}

# 初始化特征
for ntype in G_train.ntypes:
    emb = nn.Parameter(torch.Tensor(G_train.num_nodes(ntype), args.n_inp), requires_grad=False)
    nn.init.xavier_uniform_(emb)
    G_train.nodes[ntype].data['inp'] = emb

G_train = G_train.to(device)
G_test = G_test.to(device)
neg_G_test = neg_G_test.to(device)

# ================= 4. Core Functions =================
def compute_metrics(logits, labels):
    scores = torch.sigmoid(logits).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    try:
        auc_val = roc_auc_score(labels, scores)
        p, r, _ = precision_recall_curve(labels, scores)
        aupr_val = auc(r, p)
    except:
        auc_val, aupr_val = 0.5, 0.0
    preds = (scores > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return auc_val, aupr_val, acc, f1

def train_one_epoch(model, optimizer, G):
    model.train()
    target_etype = ('circ', 'circ_e_disease', 'disease')
    u, v = G.edges(etype=target_etype)
    num_edges = G.num_edges(etype=target_etype)
    
    # 随机掩码边，用于自监督/重建任务
    perm = torch.randperm(num_edges, device=device)
    mask_idx = perm[:int(num_edges * args.mask_ratio)]
    G_masked = dgl.remove_edges(G, mask_idx, etype=target_etype)
    
    pos_u, pos_v = u[mask_idx], v[mask_idx]
    n_neg = len(pos_u)
    neg_u = torch.randint(0, G.num_nodes('circ'), (n_neg,), device=device)
    neg_v = torch.randint(0, G.num_nodes('disease'), (n_neg,), device=device)
    
    heads = torch.cat([pos_u, neg_u])
    tails = torch.cat([pos_v, neg_v])
    labels = torch.cat([torch.ones(n_neg), torch.zeros(n_neg)]).to(device)
    
    # 调用模型
    logits = model(G_masked, circ_seq, dis_sem, heads, tails).squeeze()
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def save_pkl(data, name):
    """保存数据到 results 目录"""
    path = RESULTS_DIR / name
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Saved {name} to {path}")

# ================= 5. Training Routines =================
def run_single_train(G_tr, G_te, neg_G_te, fold_idx=1, capture_start=False):
    model = MaskedGraphModel(args, node_dict, edge_dict).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0
    final_roc = None
    best_model_state = None
    best_emb = None

    # === [可视化准备] 保存初始Embedding ===
    # 这里的 Embedding 仅包含原始特征（序列/语义）+ 随机初始化经过一次GNN聚合的结果
    # 用于对比训练前后的分布差异
    if capture_start:
        logging.info(f"Fold {fold_idx}: Saving Initial Embeddings (Start)...")
        model.eval()
        with torch.no_grad():
            # 使用 model.get_embedding 获取经过 GNN 层的表示
            raw_emb = model.get_embedding(G_tr, circ_seq, dis_sem)
            cpu_emb = {k: v.detach().cpu().numpy() for k, v in raw_emb.items()}
            save_pkl(cpu_emb, 'embedding_start.pkl')

    for epoch in range(1, args.n_epoch + 1):
        loss = train_one_epoch(model, optimizer, G_tr)

        if epoch % 10 == 0 or epoch == args.n_epoch:
            model.eval()
            with torch.no_grad():
                target_etype = ('circ', 'circ_e_disease', 'disease')
                pos_u, pos_v = G_te.edges(etype=target_etype)
                neg_u, neg_v = neg_G_te.edges(etype=target_etype)
                heads = torch.cat([pos_u, neg_u])
                tails = torch.cat([pos_v, neg_v])
                labels = torch.cat([torch.ones(len(pos_u)), torch.zeros(len(neg_u))]).to(device)
                
                logits = model(G_tr, circ_seq, dis_sem, heads, tails).squeeze()
                auc_val, aupr_val, acc, f1 = compute_metrics(logits, labels)

            logging.info(f"Fold {fold_idx} | Epoch {epoch:03d} | AUC: {auc_val:.4f}")

            # 记录最佳模型
            if auc_val > best_auc:
                best_auc = auc_val
                final_roc = {'labels': labels.cpu().numpy(), 'scores': torch.sigmoid(logits).cpu().numpy()}
                
                best_model_state = model.state_dict()
                # 获取当前最佳状态的 Embedding
                curr_emb = model.get_embedding(G_tr, circ_seq, dis_sem)
                best_emb = {k: v.detach().cpu().numpy() for k, v in curr_emb.items()}

    return best_auc, final_roc, best_model_state, best_emb

def run_fivefold(G):
    target_etype = ('circ', 'circ_e_disease', 'disease')
    u, v = G.edges(etype=target_etype)
    num_edges = G.num_edges(etype=target_etype)
    perm = torch.randperm(num_edges).to(device)
    fold_size = num_edges // 5

    all_roc = []
    global_best_auc = 0.0

    for i in range(5):
        logging.info(f"--- Running Fold {i+1}/5 ---")
        val_idx = perm[i * fold_size: (i + 1) * fold_size]
        G_fold_train = dgl.remove_edges(G, val_idx, etype=target_etype)
        
        pos_u, pos_v = u[val_idx], v[val_idx]
        G_fold_test = dgl.heterograph({target_etype: (pos_u, pos_v)},
                                      num_nodes_dict={nt: G.num_nodes(nt) for nt in G.ntypes}).to(device)

        n_neg = len(pos_u)
        neg_u = torch.randint(0, G.num_nodes('circ'), (n_neg,), device=device)
        neg_v = torch.randint(0, G.num_nodes('disease'), (n_neg,), device=device)
        neg_G_fold_test = dgl.heterograph({target_etype: (neg_u, neg_v)},
                                          num_nodes_dict={nt: G.num_nodes(nt) for nt in G.ntypes}).to(device)

        # 仅在第一个 Fold (i==0) 时保存初始 Embedding
        is_first = (i == 0)
        fold_auc, fold_roc, fold_model, fold_emb = run_single_train(
            G_fold_train, G_fold_test, neg_G_fold_test, 
            fold_idx=i+1, capture_start=is_first
        )

        all_roc.append(fold_roc)

        # === [可视化准备] 保存训练后(最佳) Embedding ===
        if fold_auc > global_best_auc:
            logging.info(f"🚀 New Best Model: {global_best_auc:.4f} -> {fold_auc:.4f}")
            global_best_auc = fold_auc
            save_pkl(fold_emb, 'embedding_end.pkl') # 保存最佳 Embedding 用于可视化
            
            checkpoint = {'model_state_dict': fold_model, 'args': args}
            torch.save(checkpoint, RESULTS_DIR / 'best_checkpoint.pth')
        
        torch.cuda.empty_cache()

    save_pkl(all_roc, 'roc_results.pkl')
    logging.info("Training Finished.")

if __name__ == "__main__":
    if args.fivefold:
        run_fivefold(G_train)
    else:
        # 单次运行模式下，强制保存 start embedding
        run_single_train(G_train, G_test, neg_G_test, 1, True)