import torch
import torch.nn as nn


class PageReranker(nn.Module):
    """
    MaxSim-anchored, multi-hop page re-ranker.

    Idea: ColPali's MaxSim gives a strong per-page relevance score (~77% top-1),
    but it scores every page in ISOLATION and cannot reason across pages. This
    module keeps MaxSim as an anchor and adds a small cross-page Transformer
    (the "hops") that lets every candidate page attend to the others + the
    question, then re-ranks. It is anchored on MaxSim so it starts near MaxSim's
    accuracy and only has to LEARN THE DELTA (which page MaxSim's top-1 got wrong).

    Inputs (per batch):
        page_feat          : [B, P, 2D] per-page features = [mean-pooled patches ; query-aligned evidence]
        q_mean             : [B, D]      mean-pooled question embedding
        maxsim             : [B, P]      per-page MaxSim score (z-scored per sample)
        page_padding_mask  : [B, P] bool True = padded (dummy) page
    Output:
        logits             : [B, P]      per-page selection logits (padded = -inf)
    """

    def __init__(self, config):
        super().__init__()
        D = config["model"]["embedding_dim"]
        H = config["model"]["projection_dim"]
        rc = config.get("reranker", {}) or {}
        nhead = rc.get("num_heads", 4)
        nlayers = rc.get("num_layers", 2)          # number of cross-page reasoning "hops"
        dropout = rc.get("dropout", 0.1)

        self.page_feat_dim = 2 * D                 # [page_mean ; query-aligned evidence]
        self.query_proj = nn.Linear(D, H)
        self.page_proj = nn.Linear(self.page_feat_dim, H)
        # per-page input = [projected page ; projected query ; maxsim scalar]
        self.input_proj = nn.Linear(2 * H + 1, H)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H,
            dropout=dropout, batch_first=True,
        )
        self.cross_page = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.score_head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 1))
        # Small init so the learned delta ~= 0 at the start (predictions begin at
        # MaxSim) while gradients still flow through the cross-page Transformer.
        nn.init.normal_(self.score_head[-1].weight, std=0.1)
        nn.init.zeros_(self.score_head[-1].bias)

        # Anchor weight on MaxSim. Initialized to 1.0 so that at the start
        # logits ~= maxsim  ->  the model begins at MaxSim's accuracy.
        self.maxsim_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, page_feat, q_mean, maxsim, page_padding_mask):
        B, P, _ = page_feat.shape

        q = self.query_proj(q_mean).unsqueeze(1).expand(-1, P, -1)   # [B, P, H]
        pg = self.page_proj(page_feat)                              # [B, P, H]
        feat = torch.cat([pg, q, maxsim.unsqueeze(-1)], dim=-1)     # [B, P, 2H+1]
        x = self.input_proj(feat)                                  # [B, P, H]

        # Cross-page reasoning (each page attends to the others + question context)
        x = self.cross_page(x, src_key_padding_mask=page_padding_mask)  # [B, P, H]

        learned_logit = self.score_head(x).squeeze(-1)            # [B, P]
        logits = learned_logit + self.maxsim_scale * maxsim       # anchor on MaxSim
        logits = logits.masked_fill(page_padding_mask, -1e9)
        return logits
