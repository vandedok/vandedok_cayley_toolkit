import logging
from cayleypy import CayleyGraph
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .cfg import CayleyMLCfg

logger = logging.getLogger()
logging.basicConfig(level=20)
EARLY_STOP_VERBOSE=False

def get_train_val(X, y, val_ratio: float = 0.1, stratify: bool = False):
    total_size = X.shape[0]
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    X_train, y_train, X_val, y_val = train_test_split(X, y, train_size=train_size, stratify=y.cpu().numpy() if stratify else None, shuffle=True)
    return X_train, y_train, X_val, y_val


def epoch_val(model: torch.nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, loss_fn: torch.nn.Module, batch_size: int):
    model.eval()
    total_val_loss = 0
    val_size = X_val.shape[0]
    with torch.no_grad():
        for start in range(0, val_size, batch_size):
            end = min(start + batch_size, val_size)
            xb = X_val[start:end]
            yb = y_val[start:end].float()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_val_loss += loss.item() * xb.size(0)
    return total_val_loss / val_size


def regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, batch_size):

    train_size = len(X_train)
    model.train()
    total_train_loss = 0
    for start in range(0, train_size, batch_size):
        end = min(start + batch_size, train_size)
        xb = X_train[start:end]
        yb = y_train[start:end].float()

        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)
    avg_train_loss = total_train_loss / train_size

    if X_val is not None:
        avg_val_loss = epoch_val(model, X_val, y_val, loss_fn, batch_size)
    else:
        avg_val_loss = torch.nan

    return avg_train_loss, avg_val_loss


def train_no_bellman(cfg: CayleyMLCfg, graph: CayleyGraph, model: torch.nn.Module):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.no_bellman.learning_rate)
    rw_cfg = cfg.train.random_walks
    early_stopper = EarlyStopping(patience=cfg.train.no_bellman.early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, trace_func=logger.info)
    for epoch_i in range(cfg.train.no_bellman.num_epochs):
        X, y = graph.random_walks(width=rw_cfg.width, length=rw_cfg.length, mode=rw_cfg.mode, nbt_history_depth=rw_cfg.nbt_history_depth)
        X_train, X_val, y_train, y_val = get_train_val(X, y, cfg.train.val_ratio, stratify=True)
        avg_train_loss, avg_val_loss = regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, cfg.train.no_bellman.batch_size)
        logger.info(f"Epoch {epoch_i} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            logger.info(f"Early stopper triggered at epoch {epoch_i}")
            break

def get_neighbors(X, gens):
    n_actions = gens.shape[0]
    n_states = X.shape[0]
    expanded_states = X.unsqueeze(1).expand(-1, n_actions, -1)  # [N, A, S]
    expanded_gens = gens.unsqueeze(0).expand(n_states, -1, -1)  # [N, A, S]
    neighbors = torch.gather(expanded_states, 2, expanded_gens)  # [N, A, S]
    return neighbors


def bellman_update(model, X, y, generators, batch_size, goal_state):
    batch_size = 128
    model.eval()
    X_neighbors = get_neighbors(X, generators)  # [N, A, S]
    X_neighbors_flat = X_neighbors.view(-1, X.shape[1])  # [N * A, S]
    with torch.no_grad():
        num_states = X_neighbors_flat.shape[0]
        preds_flat = torch.zeros(num_states, device=X_neighbors_flat.device)
        with torch.no_grad():
            for start in range(0, num_states, batch_size):
                end = min(start + batch_size, num_states)
                batch = X_neighbors_flat[start:end]
                preds_flat[start:end] = model(batch).squeeze()
        preds = preds_flat.view(X_neighbors.shape[0], -1)  # [N, A]
        targets = 1 + preds.min(dim=1)[0]  # [N]
        targets = torch.min(targets, y)
        targets = torch.clamp_min(targets, 1)

        is_goal = (X == goal_state).all(dim=1)
        targets[is_goal] = 0

    return targets.detach()


def train_with_bellman(cfg: CayleyMLCfg, graph: CayleyGraph, model: torch.nn.Module):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.bellman.learning_rate)
    rw_cfg = cfg.train.random_walks
    global_early_stopper = EarlyStopping(patience=cfg.train.bellman.global_early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, trace_func=logger.info)
    in_update_early_stopper = EarlyStopping(patience=cfg.train.bellman.in_update_early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, trace_func=logger.info)
    epochs_per_update = cfg.train.bellman.epochs_per_update
    
    for bellman_i in range(cfg.train.bellman.n_updates):
        logger.info(f"Bellman_update: {bellman_i}")

        X, y = graph.random_walks(width=rw_cfg.width, length=rw_cfg.length, mode=rw_cfg.mode, nbt_history_depth=rw_cfg.nbt_history_depth)
        generators = torch.tensor(graph.generators, device=graph.device)
        y_bellman = bellman_update(model, X, y, generators, cfg.train.bellman.bellman_batch_size, graph.central_state)  # In some setting goal state is not central state
        X_train, X_val, y_train, y_val = get_train_val(X, y_bellman, cfg.train.val_ratio, stratify=False)

        avg_train_loss_per_update = 0
        avg_val_loss_per_update = 0
        for epoch_i in range(epochs_per_update):
            avg_train_loss, avg_val_loss = regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, cfg.train.bellman.reg_batch_size)
            avg_train_loss_per_update += avg_train_loss
            avg_val_loss_per_update += avg_val_loss
            logger.info(f"Epoch {epoch_i} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            in_update_early_stopper(avg_val_loss, model)
            if in_update_early_stopper.early_stop:
                logger.info(f"In update early stopper triggered at epoch {epoch_i}")
                in_update_early_stopper.reset()
                break

        avg_train_loss_per_update /= epochs_per_update
        avg_val_loss_per_update /= epochs_per_update
        global_early_stopper(avg_val_loss_per_update, model)
        if in_update_early_stopper.early_stop:
            logger.info(f"Global bellman early stopper triggered at update {bellman_i}")
            break





### Taken and modified from  https://github.com/Bjarten/early-stopping-pytorch
#  commit fbd87a6135820700e27cda3448d5d54dc6fd3b0c
# MIT license

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to. Pass None to avoid saving.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.reset()

    def reset(self):
        self.early_stop = False
        self.best_val_loss = None
        self.counter = 0
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if self.path is not None:
            if self.verbose:
                self.trace_func("Saving model ...")
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss