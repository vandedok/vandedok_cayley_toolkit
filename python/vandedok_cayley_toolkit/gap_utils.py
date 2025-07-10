import numpy as np
import torch
from cayleypy import CayleyGraphDef
from .cayleypy_utils import bfs_result_to_nx_graph


def cyclic2oneline(cycle_str, n):
    """
    Taken from
    https://www.kaggle.com/code/olganikitina/permutations-cyclic-to-oneline/
    """
    one_line = list(range(n))

    cycles = cycle_str.replace(" ", "").split(")")
    for cycle in cycles:
        if not cycle:
            continue
        cycle = cycle.replace("(", "").split(",")
        cycle = [int(x) - 1 for x in cycle]
        for i in range(len(cycle) - 1):
            one_line[cycle[i]] = cycle[i + 1]
        one_line[cycle[-1]] = cycle[0]

    return one_line


def inverse_permutation(perm):
    """
    Inverts oneline permutation.Taken from:
    https://www.kaggle.com/code/alexandervc/permutations-with-numpy-tutorial#Compute-inverse-permutation
    """
    # Create an empty list to hold the inverse permutation
    inverse = [0] * len(perm)

    # Iterate over the original permutation
    for i, p in enumerate(perm):
        # Place the index at the correct position in the inverse permutation
        inverse[p] = i

    return inverse


def perm_is_id(perm):
    return np.all(perm == np.arange(len(perm)))


def add_inv_permutations(perms_dict):
    """
    Combining original and inverse permutations
    WARNING: this function doesn't check if the inverted permutations are already present
    """
    perms_dict_all = {}

    for name, perm in perms_dict.items():
        if not perm_is_id(perm):
            perms_dict_all[name] = perm
            perms_dict_all[name + "_inv"] = np.array(inverse_permutation(perm))
    return perms_dict_all


def get_permuted_set_length(perms_cyclic):
    """
    Takes the max id, the actual set length might be bigger, but why should it be?
    """
    max_idx = 0
    for p in perms_cyclic:
        ids = [int(x) for x in p.strip("(").strip(")").replace(")(", ",").split(",")]
        for idx in ids:
            if idx > max_idx:
                max_idx = idx
    return max_idx


def moves_from_twizzle_explorer_to_dict(moves_list_gap):
    return_dict = {}
    for x in moves_list_gap:
        kv = x.replace(";", "").split(":=")
        return_dict[kv[0].replace("M_", "")] = kv[1]
    return return_dict


def filter_generator_lines(gap_str):
    return [x for x in gap_str.split("\n") if ":=" in x and "Gen" not in x and "ip" not in x]


def gap_to_CayleyGraphDef(gap_generators):
    gap_generators = filter_generator_lines(gap_generators)
    gens_cyclic = moves_from_twizzle_explorer_to_dict(gap_generators)
    N = get_permuted_set_length(gens_cyclic.values())
    gens_oneline = {k: np.array(cyclic2oneline(v, N)) for k, v in gens_cyclic.items()}
    gens_oneline = add_inv_permutations(gens_oneline)
    perms_stacked = torch.from_numpy(np.stack(list(gens_oneline.values())))
    return CayleyGraphDef.create(generators=perms_stacked)
    # return cayley_graph_def
    # graph = CayleyGraph(cayley_graph_def, verbose=3, device="cpu")
