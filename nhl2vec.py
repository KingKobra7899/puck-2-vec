import torch
import torch.functional as f
import torch.nn as nn

class Puck2Vec(nn.Module):
    def __init__(self,  n_players, player_dim, transformer_dim=512 ,*args: f.Any, **kwargs: f.Any) -> None: # type: ignore
        super().__init__(*args, **kwargs)
        player_embeddings = nn.Embedding(n_players, player_dim)
        positional_embeddings = nn.Embedding(5, player_dim) #typical embeddings for each position, lw, rw, c, d, g
        pos_dim = player_dim + 2 + 1
        
        position_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim)
        )