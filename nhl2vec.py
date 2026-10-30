import torch
import torch.nn as nn
from torchview import draw_graph

class Puck2Vec(nn.Module):
    def __init__(self,  n_players, player_dim,n_frames,players_frame=11 ,transformer_dim=512 ,trans_layers=3 ,trans_heads=8, moves=12 ,*args, **kwargs) -> None: # type: ignore
        super().__init__(*args, **kwargs)
        self.n_frames = n_frames
        self.d=transformer_dim
        self.moves=moves
        self.players_frame = players_frame
        self.player_embeddings = nn.Embedding(n_players, player_dim)
        self.positional_embeddings = nn.Embedding(5, player_dim) #typical embeddings for each position, lw, rw, c, d, g
        pos_dim = player_dim + 2 + 1
        
        self.position_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim)
        )
        
        self.puck_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim)
        )
        
        self.lookahead_mlp = nn.Sequential(
            nn.Linear(pos_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim)
        )
        
        self.init_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim)
        )
        
        self.movement_head = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256,self.moves**2)
        )
        self.epv_head = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
        
        self.f = torch.arange(n_frames).repeat_interleave(players_frame)   # frame of each player token
        self.fp = torch.arange(n_frames)                                   # frame of each puck token
        self.register_buffer("frame_idx", self.f)
        self.register_buffer("puck_frame_idx", self.fp)
        fh = torch.cat([self.f, self.fp])                                       # history seq: [players | puck]
        self.register_buffer("history_mask", fh[None, :] > fh[:, None])
        
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(transformer_dim, trans_heads, batch_first=True), trans_layers)
    
    def lookahead_mask(self):
        # paper's mask: seq is, per frame, [puck, pos_1, look_1, ..., pos_K, look_K], then K start tokens
        L = self.n_frames * (2 * self.players_frame + 1)
        S = L + self.players_frame
        mask = torch.ones(S, S, dtype=torch.bool, device=self.f.device).triu(1)  # causal
        mask[:, L:] = False     # every token sees the start tokens (initial state)
        mask[L:, :L] = True     # start tokens only see each other
        return mask
    
    def forward(self, playerIds, player_pos, player_x, player_y, off, player_dxs, player_dys,puck_x,puck_y ,swap=None):
        emb = self.player_embeddings(playerIds)
        batch_size = player_x.shape[0]
        if swap is not None:                                  # [B, K] bool, rate p
            emb = torch.where(swap[..., None], self.positional_embeddings(player_pos), emb)
        emb = emb[:, None].expand(-1, self.n_frames, -1, -1)  #each input is cut to n_frames frames.
        pos_token = self.position_mlp(torch.cat([emb, torch.stack([player_x, player_y, off], -1)], -1))
        lookahead_token = self.lookahead_mlp(torch.cat([emb, torch.stack([player_x + player_dxs, player_y + player_dys,player_dxs, player_dys ,off], -1)], -1))
        init_token = self.init_mlp(torch.cat([emb[:,0], torch.stack([player_x[:,0], player_y[:,0], off[:,0]], -1)], -1))
        puck_token = self.puck_mlp(torch.stack([puck_x,puck_y], -1))
        frame = torch.cat([puck_token[:, :, None], torch.stack([pos_token, lookahead_token], 3).flatten(2, 3)], 2)  # [puck, pos_1, look_1, ...]
        seq = torch.cat([frame.flatten(1, 2), init_token], 1)

        out  = self.trans(seq, mask=self.lookahead_mask())
        L = self.n_frames * (2 * self.players_frame + 1)
        move_logits = self.movement_head(out[:, :L].reshape(-1, self.n_frames, 2 * self.players_frame + 1, self.d)[:, :, 1::2])

        n = self.n_frames * self.players_frame
        o = self.trans(torch.cat([pos_token.flatten(1, 2), puck_token], dim=1), mask=self.history_mask)[:, :n].reshape(-1, self.n_frames, self.players_frame, self.d)
        v_logits = self.epv_head(o.mean(2))
        
        return move_logits, v_logits
        
           

if __name__=="__main__":
    model = Puck2Vec(n_players=900, player_dim=32, n_frames=10,transformer_dim=256)


    B, T, K = 8, 10, 11
    inputs = (torch.randint(0, 900, (B, K)), torch.randint(0, 5, (B, K)),
            *(torch.randn(B, T, K) for _ in range(5)), torch.randn(B, T), torch.randn(B, T))
    g = draw_graph(model, input_data=inputs, depth=1, hide_inner_tensors=True, roll=True, graph_dir="TB")
    g.visual_graph.attr(label="Puck2Vec", labelloc="t", fontsize="20")
    g.visual_graph.render("model_architecture", format="png")