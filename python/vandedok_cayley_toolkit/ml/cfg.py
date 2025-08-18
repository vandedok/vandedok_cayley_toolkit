from typing import Literal
from pydantic import BaseModel

### Config ###


class RandomWalksCfg(BaseModel):
    mode: Literal["classic", "bfs", "nbt"] = "nbt"
    length: int = 256
    nbt_history_depth: int = 512
    width: int = 1000


class TrainNoBellmanCfg(BaseModel):
    batch_size: int = 1024
    learning_rate: float = 0.001
    num_epochs: int = 30
    early_stop_patience: int = 10


class TrainBellmanCfg(BaseModel):
    epochs_bfs_per_update: int = 1
    bellman_batch_size: int = 1024
    reg_batch_size: int = 1024
    n_updates: int = 20
    epochs_per_update: int = 10
    learning_rate: float = 0.001
    boundary_loss: float = 0.1
    in_update_early_stop_patience: int = 10
    global_early_stop_patience: int = 10


class TrainCfg(BaseModel):
    random_walks: RandomWalksCfg = RandomWalksCfg()
    val_ratio: float = 0.1
    no_bellman: TrainNoBellmanCfg = TrainNoBellmanCfg()
    bellman: TrainBellmanCfg = TrainBellmanCfg()


class SingleEvalCfg(BaseModel):
    n_trials: int = 100
    rw_mode: Literal["classic", "bfs", "nbt"] = "nbt"
    rw_lengths: list[int] = [16, 128, 512]
    beam_width: int = 10**3
    beam_max_steps: int = 100
    beam_mode: Literal["simple", "advanced"] = "simple"
    beam_mitm: bool = False


class ModelCfg(BaseModel):
    hidden_dims: list[int] = [512, 256]


class CayleyMLCfg(BaseModel):
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    eval: SingleEvalCfg | list[SingleEvalCfg] = SingleEvalCfg()
