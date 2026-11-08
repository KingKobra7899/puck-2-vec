from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
import torch

POSITIONS = {"L": 0, "R": 1, "C": 2, "D": 3, "G": 4}  # lw, rw, c, d, g
PLAYERS_FRAME = 11  # 5 offensive skaters, 5 defensive skaters, defensive goalie
MAX_ON_ICE = 6
MOVE_BINS = 11  # per axis, so MOVE_BINS ** 2 joint moves
MAX_SPEED_MPH = 25
FRAME_HZ = 10
MAX_MOVE = MAX_SPEED_MPH * 5280 / 3600 / FRAME_HZ  # feet per frame

def discretize_move(d):
    # equal width bins over [-MAX_MOVE, MAX_MOVE], the middle bin holds 0, anything faster lands in the end bins
    return ((d + MAX_MOVE) / (2 * MAX_MOVE) * MOVE_BINS).floor().long().clamp(0, MOVE_BINS - 1)

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame,  n_frames=50, random_window=True):
        self.n_frames = n_frames
        self.random_window = random_window
        self.sequences = []

        players = []

        for side in ["home", "away"]:
            for n in range(1, MAX_ON_ICE + 1):
                players.append(f"{side}_on_ice_{n}")

        id_cols = [f"{p}_id" for p in players]
        pos_cols = [f"{p}_pos" for p in players]
        off_cols = [f"{p}_off" for p in players]
        x_cols = [f"{p}_x" for p in players]
        y_cols = [f"{p}_y" for p in players]

        for sequence_id, rows in df.groupby("shot_id"):

                ids = rows[id_cols].to_numpy(dtype=np.float64, copy=True)
                pos = rows[pos_cols].to_numpy(dtype=object)
                off = rows[off_cols].to_numpy(dtype=bool)
                goalie = pos == "G"

                ids[goalie & off] = np.nan  # offensive goalies aren't part of the lineup
                present = ~np.isnan(ids)

                valid = (((present & off).sum(axis=1) == 5)
                         & ((present & ~off).sum(axis=1) == 6)
                         & ((present & ~off & goalie).sum(axis=1) == 1)
                         & (rows["home_n_on_ice"].to_numpy() <= MAX_ON_ICE)
                         & (rows["away_n_on_ice"].to_numpy() <= MAX_ON_ICE)
                         & rows["puck_x"].notna().to_numpy()
                         & rows["puck_y"].notna().to_numpy())

                # sort each frame by id so every column follows one player, slots shift whenever the players on ice change
                order = np.argsort(ids, axis=1)[:, :PLAYERS_FRAME]
                lineup = np.take_along_axis(ids, order, axis=1)
                x = np.take_along_axis(rows[x_cols].to_numpy(dtype=np.float32), order, axis=1)
                y = np.take_along_axis(rows[y_cols].to_numpy(dtype=np.float32), order, axis=1)
                off = np.take_along_axis(off, order, axis=1).astype(np.float32)
                pos = np.take_along_axis(pos, order, axis=1)
                puck_x = rows["puck_x"].to_numpy(dtype=np.float32)
                puck_y = rows["puck_y"].to_numpy(dtype=np.float32)
                goal = float(rows["is_goal"].iloc[0])  # one label per shot

                # a sequence is a run of valid frames with the same 11 players
                same = valid[1:] & valid[:-1] & (lineup[1:] == lineup[:-1]).all(axis=1)
                starts = np.flatnonzero(np.r_[True, ~same])
                ends = np.r_[starts[1:], len(rows)]

                for start, end in zip(starts, ends):
                    if valid[start] and end - start > n_frames:     # need n_frames + 1 for the last moves
                        self.sequences.append({
                            "player_ids": lineup[start].astype(int),                                # [players] int
                            "player_pos": torch.tensor([POSITIONS[p] for p in pos[start]], dtype=torch.long),   # [players] long
                            "x": torch.tensor(x[start:end]),            # [frames, players] float
                            "y": torch.tensor(y[start:end]),            # [frames, players] float
                            "off": torch.tensor(off[start:end]),        # [frames, players] float
                            "puck_x": torch.tensor(puck_x[start:end]),  # [frames] float
                            "puck_y": torch.tensor(puck_y[start:end]),  # [frames] float
                            "goal": torch.tensor(goal),                 # [] float, 1 if the shot was a goal
                        })

        all_ids = np.unique(np.concatenate([sequence["player_ids"] for sequence in self.sequences]))
        self.player_index = {int(pid): i for i, pid in enumerate(all_ids)}
        self.n_players = len(self.player_index)

        for sequence in self.sequences:
            sequence["player_ids"] = torch.tensor([self.player_index[int(pid)] for pid in sequence["player_ids"]], dtype=torch.long)

    def get_embedding_id(self, player_id):
        return self.player_index[int(player_id)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        total_frames = sequence["x"].shape[0]
        start = random.randint(0, total_frames - self.n_frames - 1) if self.random_window else 0
        window = slice(start, start + self.n_frames + 1)

        x, y = sequence["x"][window], sequence["y"][window]
        player_order = torch.randperm(x.shape[1])
        dx = (x[1:] - x[:-1])[:, player_order]
        dy = (y[1:] - y[:-1])[:, player_order]

        return {
            "player_ids": sequence["player_ids"][player_order],
            "player_pos": sequence["player_pos"][player_order],
            "x": x[:-1][:, player_order],
            "y": y[:-1][:, player_order],
            "off": sequence["off"][window][:-1][:, player_order],
            "dx": dx,
            "dy": dy,
            "move": discretize_move(dx) * MOVE_BINS + discretize_move(dy),  # [frames, players] long, joint class in [0, MOVE_BINS ** 2)
            "puck_x": sequence["puck_x"][window][:-1],
            "puck_y": sequence["puck_y"][window][:-1],
            "goal": sequence["goal"],
        }
