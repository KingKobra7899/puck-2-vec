import torch
import torch.functional as f
import torch.nn as nn
import visualtorch as vt
class Puck2Vec(nn.Module):
    def __init__(self,  n_players, player_dim,n_frames,players_frame=11 ,transformer_dim=512 ,moves=12 ,*args: f.Any, **kwargs: f.Any) -> None: # type: ignore
        super().__init__(*args, **kwargs)
        self.n_frames = n_frames
        self.d=transformer_dim
        self.moves=12
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
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        
        self.f = torch.arange(n_frames).repeat_interleave(players_frame)
        self.register_buffer("frame_idx", self.f)
        self.register_buffer("history_mask", self.f[None, :] > self.f[:, None]) 
        
        self.trans = nn.Transformer(d_model=transformer_dim, nhead=8)
    
    def lookahead_mask(self, rank):
        # rank: [n_players], rank[k] = k's position in the ordering
        n = self.n_frames * self.players_frame
        f = self.frame_idx
        r = rank.repeat(self.n_frame)

        past = self.f[None, :] < self.f[:, None]
        upto = self.f[None, :] <= self.f[:, None]
        same = self.f[None, :] == self.f[:, None]
        before = r[None, :] < r[:, None]
        not_after = r[None, :] <= r[:, None]

        Z, U, Q = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
        allow = torch.zeros(3 * n, 3 * n, dtype=torch.bool, device=self.f.device)
        allow[Z, Z] = upto
        allow[U, Z] = upto
        allow[U, U] = past | (same & not_after)
        allow[Q, Z] = upto
        allow[Q, U] = past | (same & before)
        allow[Q, Q] = torch.eye(n, dtype=torch.bool, device=self.f.device)
        return ~allow
    
    def forward(self, playerIds, player_pos, player_x, player_y, off, player_dxs, player_dys,puck_x,puck_y ,swap=None):
        emb = self.player_embeddings(playerIds)
        batch_size = player_x.shape[0]
        if swap is not None:                                  # [B, K] bool, rate p
            emb = torch.where(swap[..., None], self.positional_embeddings(player_pos), emb)
        emb = emb[:, None].expand(-1, self.n_frames, -1, -1)  #each input is cut to n_frames frames.
        pos_token = self.position_mlp(torch.cat([emb, player_x, player_y, off]))
        lookahead_token = self.position_mlp(torch.cat([emb, player_x + player_dxs, player_y + player_dys,player_dxs, player_dys ,off]))
        init_token = self.position_mlp(torch.cat([emb, player_x[:,0], player_y[:,0], off[:,0]]))
    
        seq = torch.cat([pos_token, lookahead_token,init_token])
        
        order = torch.randperm(self.players_frame)                  # training: random ordering
        rank = torch.empty_like(order)
        rank[order] = torch.arange(self.players_frame) 
        
        out  = self.trans(seq, mask=self.lookahead_mask(rank))
        move_logits = self.movement_head(out[:, 2 * (self.n_frames * self.players_frame):].view(-1, self.n_frames, 10, self.d))
        
        o = self.trans(pos_token, mask=self.history_mask).view(-1, self.n_frames, 10, self.d)
        v_logits = self.epv_head(o.mean(2))
        
        return move_logits, v_logits
        
           

if __name__=="__main__":
    model = Puck2Vec(n_players=900, player_dim=32, n_frames=10,transformer_dim=256)
    vt.render(model, input_shape=(8,10,10),to_file='model_architecture.png')