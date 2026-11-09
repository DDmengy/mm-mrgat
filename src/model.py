import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
# [核心优化] 引入 DGL 官方优化的层
# 新增引入 GraphConv 用于消融实验对比
from dgl.nn import HeteroGraphConv, GATConv, GraphConv


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dict, n_heads, dropout, use_gat=True):
        """
        use_gat: bool, True 表示使用注意力机制 (GAT), False 表示使用普通卷积 (GCN)
        """
        super(GNNLayer, self).__init__()
        self.n_heads = n_heads
        self.use_gat = use_gat
        
        # GAT 需要除以头数，GCN 不需要
        self.d_k = out_dim // n_heads if use_gat else out_dim

        if self.use_gat:
            # === 使用 GAT (Attention) ===
            self.conv = HeteroGraphConv({
                etype: GATConv(
                    in_feats=in_dim,
                    out_feats=self.d_k,  # 每个头的维度
                    num_heads=n_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
                for etype in edge_dict
            }, aggregate='sum')
        else:
            # === 使用 GCN (No Attention) 用于消融实验 ===
            self.conv = HeteroGraphConv({
                etype: GraphConv(
                    in_feats=in_dim,
                    out_feats=out_dim, # GCN 直接输出目标维度
                    norm='both',       # 标准 GCN 归一化
                    weight=True,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
                for etype in edge_dict
            }, aggregate='sum')

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, h):
        # 1. DGL 卷积
        h_out = self.conv(G, h)

        res = {}
        for ntype in G.ntypes:
            if ntype in h_out:
                feat = h_out[ntype]
                
                # 2. 维度处理
                # GAT 输出 shape: [N, num_heads, head_dim] -> 需要 flatten
                # GCN 输出 shape: [N, out_dim] -> 不需要 flatten
                if self.use_gat:
                    feat = feat.flatten(1)

                # 3. 残差连接 + 归一化
                # 如果维度对齐则相加 (防止第一层维度变化导致无法相加)
                if feat.shape[-1] == h[ntype].shape[-1]:
                    feat = feat + h[ntype]

                feat = self.norm(feat)
                res[ntype] = self.dropout(feat)
            else:
                # 孤立节点保持原样
                res[ntype] = h[ntype]
        return res


class MaskedGraphModel(nn.Module):
    def __init__(self, args, node_dict, edge_dict):
        super(MaskedGraphModel, self).__init__()
        self.node_dict = node_dict
        self.args = args

        # 特征适应层 (Input -> Hidden)
        self.adapt_ws = nn.ModuleList([
            nn.Linear(args.n_inp, args.n_hid) for _ in range(len(node_dict))
        ])

        # 判断是否使用 GCN (通过 args 获取，默认为 False 即使用 GAT)
        # getattr 是一种安全的写法，防止 args 里没有这个参数报错
        use_gcn = getattr(args, 'use_gcn', False)
        use_gat = not use_gcn

        # GNN Encoder
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(
                GNNLayer(
                    in_dim=args.n_hid,
                    out_dim=args.n_hid,
                    edge_dict=edge_dict,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    use_gat=use_gat  # === 传入开关 ===
                )
            )

        # Decoder (Link Prediction MLP)
        self.decoder = nn.Sequential(
            nn.Linear(args.n_hid * 2, args.n_hid),
            nn.ReLU(),
            nn.Linear(args.n_hid, 1)
        )

    def get_embedding(self, G, circ_seq, dis_sem):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            base = G.nodes[ntype].data['inp']

            # 多模态特征融合 (简单相加，确保维度匹配)
            if ntype == 'circ' and circ_seq is not None:
                feat_dim = min(base.shape[1], circ_seq.shape[1])
                feat = base.clone()
                feat[:, :feat_dim] += circ_seq[:, :feat_dim]
            elif ntype == 'disease' and dis_sem is not None:
                feat_dim = min(base.shape[1], dis_sem.shape[1])
                feat = base.clone()
                feat[:, :feat_dim] += dis_sem[:, :feat_dim]
            else:
                feat = base

            h[ntype] = self.adapt_ws[n_id](feat)

        for layer in self.layers:
            h = layer(G, h)
        return h

    def forward(self, G, circ_seq, dis_sem, heads, tails):
        h = self.get_embedding(G, circ_seq, dis_sem)

        # 提取边特征
        h_src = h['circ'][heads]
        h_dst = h['disease'][tails]

        edge_feat = torch.cat([h_src, h_dst], dim=-1)
        return self.decoder(edge_feat)