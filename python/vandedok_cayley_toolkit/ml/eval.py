from cayleypy import CayleyGraph, Predictor
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .cfg import CayleyMLCfg, SingleEvalCfg


def measure_success(
    graph,
    predictors,
    beam_width,
    rw_length,
    n_trials,
    max_steps,
    beam_mode="simple",
    bfs_result_for_mitm=None,
):
    success_counts = {name: 0 for name in predictors.keys()}

    X, y = graph.random_walks(width=n_trials, length=rw_length, mode="nbt")

    X_trajs = X.view(rw_length, n_trials, -1)
    # y_trajs = y.view(rw_length, n_trials)
    for trial_i in range(n_trials):

        start_state = X_trajs[-1, trial_i].cpu().numpy()
        for predictor_name, predictor in predictors.items():

            graph.free_memory()
            result = graph.beam_search(
                start_state=start_state,
                beam_width=beam_width,
                max_steps=max_steps,
                predictor=predictor,
                beam_mode=beam_mode,
                bfs_result_for_mitm=bfs_result_for_mitm,
            )
            success_counts[predictor_name] += result.path_found
    return {name: count for name, count in success_counts.items()}


def evaluate(graph: CayleyGraph, eval_cfg: SingleEvalCfg, predictors: dict[str, Predictor] = {}, bfs_result_for_mitm=None):

    if eval_cfg.beam_mitm and bfs_result_for_mitm is None:
        raise ValueError("If eval_cfg.beam_mitm, bfs_result_for_mitm has to be provided")
    # predictor = Predictor(graph, model)
    success_counts = {}

    for rw_length in tqdm(eval_cfg.rw_lengths, desc=f"RW lengths", leave=False):
        success_rates_predictors = measure_success(
            graph,
            predictors,
            eval_cfg.beam_width,
            rw_length,
            eval_cfg.n_trials,
            eval_cfg.beam_max_steps,
            bfs_result_for_mitm=bfs_result_for_mitm if eval_cfg.beam_mitm else None,
            beam_mode=eval_cfg.beam_mode,
        )
        for predictor_name in predictors.keys():
            if predictor_name not in success_counts:
                success_counts[predictor_name] = []
            success_counts[predictor_name].append(success_rates_predictors[predictor_name])
        # for predictor_name in predictors.keys():
        #     success_rates[beam_w][predictor_name].append(success_rates_predictors[predictor_name])
    return success_counts, {k: np.array(v) / eval_cfg.n_trials for k, v in success_counts.items()}


from statsmodels.stats.proportion import proportion_confint


def show_eval_plots(success_counts, rw_lengths, beam_width, max_steps, n_trials, conf_alpha=0.95, puzzle_name=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for predictor_name, counts in success_counts.items():
        counts_plot = np.array(counts)
        ax.plot(rw_lengths, counts_plot / n_trials, label=predictor_name, marker="o")

        if conf_alpha is not None:
            lower, upper = proportion_confint(count=counts_plot, nobs=n_trials, alpha=0.05, method="wilson")
            ax.fill_between(x=rw_lengths, y1=lower, y2=upper, label=predictor_name, alpha=0.2)

    ax.set_xticks(np.arange(0, max(rw_lengths) + 1, 50))  # Set minor ticks for x-axis
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set minor ticks for y-axis
    ax.set_xlabel("Random Walk Length")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"{puzzle_name}, beam size {beam_width}, max_steps: {max_steps}")
    ax.legend()
    # ax.set_xscale('log')
    ax.grid(which="both")
    ax.set_xlim(left=1, right=max(rw_lengths) + 1)
    ax.set_ylim(bottom=-0.05, top=1.05)

    return fig, ax
