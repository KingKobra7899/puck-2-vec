import torch
import torch.nn as nn
from torchview import draw_graph


class Puck2Vec(nn.Module):
    def __init__(self, player_dim, n_frames, players_frame=11, transformer_dim=512, trans_layers=3, trans_heads=8, moves=11, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self.n_frames = n_frames
        self.d = transformer_dim
        self.moves = moves
        self.players_frame = players_frame

        self.positional_embeddings = nn.Embedding(
            5, player_dim
        )  # typical embeddings for each position, lw, rw, c, d, g
        pos_dim = player_dim + 2 + 3  # emb, x, y, [off, puck_x, puck_y]

        self.position_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim),
        )

        self.lookahead_mlp = nn.Sequential(
            nn.Linear(pos_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim),
        )

        self.init_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_dim),
        )

        self.movement_head = nn.Sequential(
            nn.Linear(transformer_dim, 256), nn.ReLU(), nn.Linear(256, self.moves**2)
        )

        self.register_buffer("mask", self.lookahead_mask())
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(transformer_dim, trans_heads, batch_first=True),
            trans_layers,
        )

    def lookahead_mask(self):
        # paper's mask: seq is K init tokens, then per frame [pos_1, look_1, ..., pos_K, look_K]
        K = self.players_frame
        S = K + 2 * self.n_frames * K
        mask = torch.ones(S, S, dtype=torch.bool).triu(
            1
        )  # causal: pos_k sees pos_<=k, look_<k; look_k sees pos_<=k, look_<=k
        mask[:K, :K] = (
            False  # init tokens only see each other; everything else sees them
        )
        return mask

    def encode(
        self,
        player_pos,
        player_x,
        player_y,
        off,
        player_dxs,
        player_dys,
        puck_x,
        puck_y,
    ):
        emb = self.positional_embeddings(player_pos)  # e(p_k)
        emb = emb[:, None].expand(-1, self.n_frames, -1, -1)
        puck = torch.stack([puck_x, puck_y], -1)[:, :, None].expand(
            -1, -1, self.players_frame, -1
        )  # puck location, shared by every player in the frame
        # off + puck = context h_{t,k}
        pos_token = self.position_mlp(
            torch.cat([emb, torch.stack([player_x, player_y, off], -1), puck], -1)
        )  # z_{t,k}
        lookahead_token = self.lookahead_mlp(
            torch.cat(
                [
                    emb,
                    torch.stack(
                        [
                            player_x + player_dxs,
                            player_y + player_dys,
                            player_dxs,
                            player_dys,
                            off,
                        ],
                        -1,
                    ),
                    puck,
                ],
                -1,
            )
        )  # u_{t,k}
        init_token = self.init_mlp(
            torch.cat(
                [
                    emb[:, 0],
                    torch.stack([player_x[:, 0], player_y[:, 0], off[:, 0]], -1),
                    puck[:, 0],
                ],
                -1,
            )
        )  # r_k
        frame = torch.stack([pos_token, lookahead_token], 3).flatten(
            2, 3
        )  # [pos_1, look_1, ...]
        seq = torch.cat([init_token, frame.flatten(1, 2)], 1)  # K + 2TK tokens

        out = self.trans(seq, mask=self.mask)
        K = self.players_frame
        return out[:, K:].reshape(
            -1, self.n_frames, K, 2, self.d
        )  # [B, T, K, pos|look, d]

    def get_rink_state(
        self,
        playerIds,
        player_pos,
        player_x,
        player_y,
        off,
        player_dxs,
        player_dys,
        puck_x,
        puck_y,
        pool="mean",
    ):
        # every player's pos_token at frame t = the model's read of the board at t
        pos_out = self.encode(
            player_pos, player_x, player_y, off, player_dxs, player_dys, puck_x, puck_y
        )[
            :, :, :, 0
        ]  # [B, T, K, d]
        if pool == "mean":
            return pos_out.mean(2)  # [B, T, d]
        if pool == "concat":
            return pos_out.flatten(2, 3)  # [B, T, K*d], players in input order
        return pos_out  # [B, T, K, d]

    def forward(
        self,
        playerIds,
        player_pos,
        player_x,
        player_y,
        off,
        player_dxs,
        player_dys,
        puck_x,
        puck_y,
    ):
        pos_out = self.encode(
            player_pos, player_x, player_y, off, player_dxs, player_dys, puck_x, puck_y
        )[:, :, :, 0]
        move_logits = self.movement_head(pos_out)  # read moves off each pos_token

        return move_logits


if __name__ == "__main__":
    model = Puck2Vec(player_dim=32, n_frames=10, transformer_dim=256)

    B, T, K = 8, 10, 11
    inputs = (
        torch.randint(0, 900, (B, K)),
        torch.randint(0, 5, (B, K)),
        *(torch.randn(B, T, K) for _ in range(5)),
        torch.randn(B, T),
        torch.randn(B, T),
    )
    g = draw_graph(
        model,
        input_data=inputs,
        depth=1,
        hide_inner_tensors=True,
        roll=True,
        graph_dir="TB",
    )
    g.visual_graph.attr(label="Puck2Vec", labelloc="t", fontsize="20")
    g.visual_graph.render("model_architecture", format="png")
