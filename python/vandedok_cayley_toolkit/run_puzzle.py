# from vandedok_cayley_toolkit import (
#     read_txt,
#     write_txt,
#     read_json,
#     write_json,
#     show_spectrum,
#     gap_to_CayleyGraphDef,
#     kamada_kawai_layered_layout,
#     draw_graph_with_nx,
#     bfs_result_to_nx_graph,
#     show_multiple_spectra,
#     wigner_semicircle,
#     normalize_adj_matrix,
# )
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from cayleypy import CayleyGraph, BfsResult
import networkx as nx
from .general_utils import write_json, write_txt
from .gap_utils import gap_to_CayleyGraphDef
from .cayleypy_utils import bfs_result_to_nx_graph
from .graphs_utils import kamada_kawai_layered_layout, layered_sequential_FD_layout, draw_graph_with_nx
from .graph_spectrum import show_spectrum, show_multiple_spectra, normalize_adj_matrix


def draw_puzzle_graphs(
    output_dir: str,
    puzzle_name: str,
    gap_generators: str,
    max_kamada_kawai_diameter: int,
    kamada_kawai_layers_weights: list = [10, 10, 70],
    max_layered_fd_diameter: int = 3,
    layered_fd_layers_weights: list = [200, 400, 1000],
    layered_fd_k_modifiers_no_layers: list = [1, 1, 1],
    layered_fd_k_modifiers_layers: list = [1, 1, 1],
    max_fd_diameter: int = 4,
    fd_k_modifiers: list = [1, 1, 1, 1],
    redo_if_exists: bool = True,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    # gap_generators = read_txt("../vanedok_cayley_toolkit/puzzle_generators/pyraminx_tertraminx/tetraminx.gap")
    print(gap_generators)
    graph_def = gap_to_CayleyGraphDef(gap_generators)
    graph = CayleyGraph(graph_def, verbose=3, device="cpu")
    plt.rcParams.update({"font.size": 10})

    ######################
    ### Drawing graphs ###

    # Visualizing first few layes. Can take a while, alos when its too much vertices the result is messy
    # Trying different numbers of layers and layouts might be a good idea
    # max_layer == 4 gives messy result, so dropping it

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    bfs_result = graph.bfs(
        return_all_edges=True,
        return_all_hashes=True,
        max_layer_size_to_store=None,
        max_diameter=max(max_kamada_kawai_diameter, max_layered_fd_diameter, max_fd_diameter),
    )

    layout = "layered_kamada_kawai"
    layout_dir = graphs_dir / f"{layout}"
    layout_dir.mkdir(exist_ok=True)
    # layers_weights = kamada_kawai_layers_weights[: max_kamada_kawai_diameter + 1]
    graph_max_layers = list(range(1, max_kamada_kawai_diameter + 1))
    for max_layer, layers_weight in zip(graph_max_layers, kamada_kawai_layers_weights):
        figure_path = layout_dir / f"{puzzle_name}_max_layer_{max_layer}_layout_{layout}"
        if redo_if_exists or not figure_path.exists():
            nx_graph = bfs_result_to_nx_graph(bfs_result, max_layer=max_layer)
            nodes_layers = np.array([x[1]["layer_"] for x in nx_graph.nodes(data=True)])
            pos = kamada_kawai_layered_layout(nx_graph, nodes_layers=nodes_layers, layers_weight=layers_weight, store_pos_as="pos_")
            fig, ax = draw_graph_with_nx(nx_graph, pos=pos, draw_layers_circles=True)
            ax.set_title(f"{puzzle_name}, first {max_layer} layers, {layout} layout")
            fig.savefig(figure_path)
            plt.show()
        else:
            img = plt.imread(figure_path)
            plt.imshow(img)

    layout = "layered_fruchterman_reingold"
    layout_dir = graphs_dir / f"{layout}"
    layout_dir.mkdir(exist_ok=True)
    graph_max_layers = list(range(1, max_layered_fd_diameter + 1))
    for max_layer in graph_max_layers:
        figure_path = layout_dir / f"{puzzle_name}_max_layer_{max_layer}_layout_{layout}"
        if redo_if_exists or not figure_path.exists():
            nx_graph = bfs_result_to_nx_graph(bfs_result, max_layer=max_layer)
            pos = layered_sequential_FD_layout(
                nx_graph,
                layers_weights=layered_fd_layers_weights[:max_layer],
                k_modifiers_no_layers=layered_fd_k_modifiers_no_layers[:max_layer],
                k_modifiers_layers=layered_fd_k_modifiers_layers[:max_layer],
            )
            fig, ax = draw_graph_with_nx(nx_graph, pos=pos, draw_layers_circles=True)
            ax.set_title(f"{puzzle_name}, first {max_layer} layers, {layout} layout")
            fig.savefig(figure_path)
            plt.show()
        else:
            img = plt.imread(figure_path)
            plt.imshow(img)

    layout = "fruchterman_reingold"
    layout_dir = graphs_dir / f"{layout}"
    layout_dir.mkdir(exist_ok=True)
    graph_max_layers = list(range(1, max_fd_diameter + 1))
    for max_layer, k_modifier in zip(graph_max_layers, fd_k_modifiers):
        figure_path = layout_dir / f"{puzzle_name}_max_layer_{max_layer}_layout_{layout}"
        if redo_if_exists or not figure_path.exists():
            nx_graph = bfs_result_to_nx_graph(bfs_result, max_layer=max_layer)

            pos = nx.spring_layout(nx_graph, k=k_modifier / np.sqrt(len(nx_graph.nodes)))
            fig, ax = draw_graph_with_nx(nx_graph, pos=pos)
            ax.set_title(f"{puzzle_name}, first {max_layer} layers, {layout} layout")
            fig.savefig(figure_path)
            plt.show()
        else:
            img = plt.imread(figure_path)
            plt.imshow(img)


def run_puzzle_analysis(
    output_dir: str,
    puzzle_name: str,
    gap_generators: str,
    max_bfs_diameter: int = 1000000,
    max_spectrum_diameter: int = 4,
    sparse_matrix_from_diam: int = 6,
    spectrum_bins: int = 50,
    redo_if_exists: bool = False,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    # gap_generators = read_txt("../vanedok_cayley_toolkit/puzzle_generators/pyraminx_tertraminx/tetraminx.gap")
    print(gap_generators)
    graph_def = gap_to_CayleyGraphDef(gap_generators)
    graph = CayleyGraph(graph_def, verbose=3, device="cpu")


   
    # updating global plt fontsize from now on
    plt.rcParams.update({"font.size": 20})

    ######################################
    ### Running bfs up to max_bfs_diam ###

    bfs_result_dir = output_dir / "bfs_results"
    bfs_result_dir.mkdir(exist_ok=True)
    bfs_result_path = bfs_result_dir / f"{puzzle_name}_bfs_result.h5"
    # Getting all layres sizes. Might not get till the end -- try limiting the  maximum layer in this case

    if redo_if_exists or not bfs_result_path.exists():
        bfs_result = graph.bfs(max_layer_size_to_store=None, max_diameter=max_bfs_diameter)
        bfs_result.save(bfs_result_path)
    else:
        bfs_result = BfsResult.load(bfs_result_path)

    print("layer_sizes:", bfs_result.layer_sizes)
    write_json(bfs_result_dir / f"cayleypy_layers_{puzzle_name}.json", bfs_result.layer_sizes)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(f"Growth {puzzle_name}")
    x = range(len(bfs_result.layer_sizes))
    y = bfs_result.layer_sizes
    ax.plot(x, y, label=f"{puzzle_name}")
    ax.scatter(x, y, label=f"{puzzle_name}")
    for i in range(1, 12):
        plt.text(x[i], y[i], f"{y[i]:.1e}", fontsize=20, ha="right", va="bottom", fontweight="bold")
    for i in range(12, len(y)):
        plt.text(x[i], y[i], f"{y[i]:.1e}", fontsize=20, ha="left", va="bottom", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle="--")
    ax.set_xlim(left=0, right=len(y) + 1)
    ax.set_ylim(bottom=1)
    ax.set_ylabel("Number of States")
    ax.set_xlabel("Distance")
    fig.savefig(bfs_result_dir / "growth.png")

    #######################
    ### Drawing spectra ###

    spectrum_dir = output_dir / "spectrum"
    eigen_str_dir = spectrum_dir / "eigenvalues_strings"
    eigen_csv_dir = spectrum_dir / "csv"
    histgrams_dir = spectrum_dir / "histograms"

    for d in [spectrum_dir, eigen_str_dir, eigen_csv_dir, histgrams_dir]:
        d.mkdir(exist_ok=True, parents=True)

    spectrum_diams = list(range(1, max_spectrum_diameter + 1))
    for max_diam in spectrum_diams:

        eigenvals_txt_path = eigen_str_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.txt"

        if redo_if_exists or not eigenvals_txt_path.exists():

            bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_layer_size_to_store=None, max_diameter=max_diam)
            if max_diam < sparse_matrix_from_diam:
                print("Using numpy for getting eigenvalues")
                adj_matr = bfs_result.adjacency_matrix()
                eigenvals = np.linalg.eigvalsh(adj_matr)
            else:
                print("Using scipy.sparse for getting eigenvalues")
                adj_matr = bfs_result.adjacency_matrix_sparse()
                eigenvals = scipy.sparse.linalg.eigsh(adj_matr, k=adj_matr.shape[0] - 1, return_eigenvectors=False, which="BE")
            eigenvals = np.sort(eigenvals)
            eigenvals_rounded = np.round(eigenvals, decimals=2)
            eigenvals_rounded_unique, eigenvals_counts = np.unique(eigenvals_rounded, return_counts=True)
            eigenvals_str = "[" + ", ".join([f"{x}^{y}" for x, y in zip(eigenvals_rounded_unique, eigenvals_counts)]) + "]"
            df_eignevals = pd.DataFrame.from_dict({"eigenvalue": eigenvals_rounded_unique, "counts": eigenvals_counts})
            fig, ax = show_spectrum(eigenvals_rounded, figsize=(20, 10), bins=spectrum_bins)
            ax.set_title(f"{puzzle_name} spectrum, first {max_diam} layers")
            plt.show()
            print(f"****************\nEigenvalues and counts ({max_diam} layers):\n{eigenvals_str}\n****************")

            fig.savefig(histgrams_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.png")
            write_txt(eigenvals_txt_path, eigenvals_str)
            df_eignevals.to_csv(eigen_csv_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.csv")

    spectrums = []
    for max_diam in spectrum_diams:

        eigenvals_txt_path = eigen_str_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.txt"

        if redo_if_exists or not eigenvals_txt_path.exists():
            bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_layer_size_to_store=None, max_diameter=max_diam)
            if max_diam < sparse_matrix_from_diam:
                print("Using numpy for getting eigenvalues")
                adj_matr = bfs_result.adjacency_matrix()
                adj_matr = normalize_adj_matrix(adj_matr)
                eigenvals = np.linalg.eigvalsh(adj_matr)
            else:
                raise Exception("To be implemented for sparse matrices")
                print("Using scipy.sparse for getting eigenvalues")
                adj_matr = bfs_result.adjacency_matrix_sparse()
                eigenvals = scipy.sparse.linalg.eigsh(adj_matr, k=adj_matr.shape[0] - 1, return_eigenvectors=False, which="BE")

            eigenvals = np.sort(eigenvals)
            eigenvals_rounded = np.round(eigenvals, decimals=2)
            eigenvals_rounded_unique, eigenvals_counts = np.unique(eigenvals_rounded, return_counts=True)
            eigenvals_str = "[" + ", ".join([f"{x}^{y}" for x, y in zip(eigenvals_rounded_unique, eigenvals_counts)]) + "]"
            df_eignevals = pd.DataFrame.from_dict({"eigenvalue": eigenvals_rounded_unique, "counts": eigenvals_counts})
            fig, ax = show_spectrum(eigenvals_rounded, figsize=(20, 10), bins=spectrum_bins)
            ax.set_title(f"{puzzle_name} spectrum (normalized), first {max_diam} layers")
            plt.show()
            print(f"****************\nEigenvalues and counts ({max_diam} layers):\n{eigenvals_str}\n****************")

            fig.savefig(histgrams_dir / f"{puzzle_name}_spectrum_norm_{max_diam}_layers.png")
            write_txt(eigen_str_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.txt", eigenvals_str)
            df_eignevals.to_csv(eigen_csv_dir / f"{puzzle_name}_spectrum_{max_diam}_layers.csv")
            spectrums.append(eigenvals)

    fix, ax = show_multiple_spectra(spectrums)
    fig.savefig(histgrams_dir / f"{puzzle_name}_multiple_layers_spectrums_log.png")
    fix, ax = show_multiple_spectra(spectrums, logscale=False)
    fig.savefig(histgrams_dir / f"{puzzle_name}_multiple_layers_spectrums.png")
