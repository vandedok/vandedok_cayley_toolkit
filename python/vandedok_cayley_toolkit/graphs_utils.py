import networkx as nx
import numpy as np
import scipy as sp
from networkx.drawing.layout import (
    random_layout,
    circular_layout,
    rescale_layout,
    _kamada_kawai_solve,
    _process_params,
    np_random_state,
    fruchterman_reingold_layout,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
from .matplotlib_utils import get_fig_ax, lighten
from .cayleypy_utils import bfs_result_to_nx_graph


def layered_layout_bands(nodes_layers):
    _, layers_counts = np.unique(nodes_layers, return_counts=True)

    # vols = np.log(np.e-1+layers_counts) * layers_counts
    # vols = np.pow(layers_counts, -0.5) * layers_counts
    vols = layers_counts
    vols_cumsum = np.cumsum(vols)
    radii = np.pow(vols_cumsum / np.pi, 0.5)
    # radii[0] = 0.01
    radii = np.append(radii[::-1], [0])[::-1]
    max_layer = np.max(nodes_layers)
    radii = np.arange(max_layer + 1)
    return np.stack([radii, radii + 0.5]).T


def kamada_kawai_layered_layout(
    G,
    dist=None,
    pos=None,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    store_pos_as=None,
    nodes_layers=None,
    layers_weight=10,
):
    """Position nodes using Kamada-Kawai path-length cost-function and extra potential to ensure the layers will be separated.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G. All nodes must have 'layer_' attribute

    dist : dict (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    store_pos_as : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name, which can be accessed with
        ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> from pprint import pprint
    >>> G = nx.path_graph(4)
    >>> pos = nx.kamada_kawai_layout(G)
    >>> # suppress the returned dict and store on the graph directly
    >>> _ = nx.kamada_kawai_layout(G, store_pos_as="pos")
    >>> pprint(nx.get_node_attributes(G, "pos"))
    {0: array([0.99996577, 0.99366857]),
     1: array([0.32913544, 0.33543827]),
     2: array([-0.33544334, -0.32910684]),
     3: array([-0.99365787, -1.        ])}
    """
    import numpy as np

    assert all("layer_" in data for _, data in G.nodes(data=True)), "Some nodes don't have 'layer_' attribute"

    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 3:
            pos = random_layout(G, dim=dim)
        elif dim == 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = dict(zip(G, np.linspace(0, 1, len(G))))
    pos_arr = np.array([pos[n] for n in G])

    # nodes_layers = np.array([x[1]["layer_"] for x in G.nodes(data=True)])
    if nodes_layers is None:
        raise ValueError("Nodes layers are requred for layered Kamada-Kawaii layout")
    bands = layered_layout_bands(nodes_layers)
    bands = bands[nodes_layers]

    pos = _kamada_kawai_solve(dist_mtx, pos_arr=pos_arr, dim=dim)
    pos = _kamada_kawai_layered_solve(dist_mtx, pos_arr=pos, dim=dim, bands=bands, layers_weight=layers_weight)

    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))

    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos


def _kamada_kawai_layered_solve(dist_mtx, pos_arr, dim, bands, layers_weight=10.0):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np
    import scipy as sp

    meanwt = 1e-3
    costargs = (
        np,
        1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3),
        meanwt,
        dim,
        bands,
        layers_weight,
    )

    optresult = sp.optimize.minimize(
        _kamada_kawai_layered_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, dim))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _kamada_kawai_layered_costfn(pos_vec, np, invdist, meanweight, dim, bands, layers_weight):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum("ij,ij,ijk->jk", invdist, offset, direction)

    # # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    # Additional term to move the layers to the corresponding bands

    pos_radii = np.sqrt(np.sum(np.pow(pos_arr, 2), axis=1)).reshape(-1, 1)

    delta_lower = pos_radii - bands[:, 0:1]
    delta_higher = pos_radii - bands[:, 1:]
    mask_lower = delta_lower < 0
    mask_higher = delta_higher > 0
    delta_lower = delta_lower * mask_lower
    delta_higher = delta_higher * mask_higher

    cost_layers = np.pow(delta_lower, 2) + np.pow(delta_higher, 2)
    grad_layers = (2 * delta_lower + 2 * delta_higher) * pos_arr / pos_radii

    cost += layers_weight * cost_layers.sum()
    grad += layers_weight * grad_layers

    return (cost, grad.ravel())


def get_layers_circles(nodes_layers, nodes_positions, cmap):
    max_layer = nodes_layers.max()
    radii = np.sqrt((nodes_positions**2).sum(axis=1))
    layers_mins = np.ones(max_layer + 1) * 1000
    layers_maxs = np.zeros(max_layer + 1)

    for layer, radius in zip(nodes_layers, radii):
        if layers_mins[layer] > radius:
            layers_mins[layer] = radius
        if layers_maxs[layer] < radius:
            layers_maxs[layer] = radius

    layers_mins = np.concatenate([layers_mins, [layers_maxs[-1] * 1.1]])
    layers_maxs = np.concatenate([[0], layers_maxs])
    circles_radii = (layers_maxs + layers_mins) / 2

    return [
        plt.Circle(
            xy=(0, 0),
            radius=x,
            facecolor=lighten(cmap(i), 0.1),
            linewidth=1,
            edgecolor="black",
            linestyle="dotted",
        )
        for i, x in enumerate(circles_radii[1:])
    ]


def draw_graph_with_nx(
    nx_graph,
    pos,
    cmap=colormaps["Set1"],
    node_alpha=1.0,
    node_size=30,
    node_linewidth=1.0,
    edge_alpha=0.1,
    edge_width=1.0,
    draw_layers_circles=False,
    figsize=(10, 10),
    fig=None,
    ax=None,
):
    # pos = nx.spring_layout(nx_graph)
    # pos = getattr(nx, f"{layout}_layout")(nx_graph)

    node_colors = [cmap([nx_graph.nodes[node]["layer_"]]) for node in nx_graph.nodes]
    layers_ids = [nx_graph.nodes[node]["layer_"] for node in nx_graph.nodes]
    # nx.draw(nx_graph, pos, node_color=node_colors, edge_color =edge_color,  node_size=10, width=0.5)

    fig, ax = get_fig_ax(fig, ax, figsize=figsize)

    if draw_layers_circles:
        nodes_layers = np.array([int(x[1]["layer_"]) for x in nx_graph.nodes(data=True)])
        nodes_positions = np.array([x[1]["pos_"] for x in nx_graph.nodes(data=True)])
        circles = get_layers_circles(nodes_layers, nodes_positions, cmap)
        for circle in circles[::-1]:
            ax.add_patch(circle)

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=node_alpha,
        linewidths=node_linewidth,
        edgecolors="black",
    )
    nx.draw_networkx_edges(nx_graph, pos, alpha=edge_alpha, width=edge_width, ax=ax)
    legend_elements = [Patch(facecolor=cmap(x), label=x) for x in set(layers_ids)]
    ax.legend(handles=legend_elements, title="Layer")
    return fig, ax


@np_random_state(10)
def spring_layered_layout(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    store_pos_as=None,
    *,
    # method="auto",
    gravity=1.0,
    nodes_layers=None,
    layers_weight=10.0,
):
    """Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.

    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.
        Nodes not in ``G.nodes`` are ignored.
        ValueError raised if `fixed` specified and `pos` not.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  Larger means a stronger attractive force.
        If None, then all edge weights are 1.

    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    store_pos_as : str, default None
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name, which can be accessed with
        ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.

    gravity: float optional (default=1.0)
        Used only for the method='energy'.
        The positive coefficient of gravitational forces per connected component.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> from pprint import pprint
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)
    >>> # suppress the returned dict and store on the graph directly
    >>> _ = nx.spring_layout(G, seed=123, store_pos_as="pos")
    >>> pprint(nx.get_node_attributes(G, "pos"))
    {0: array([-0.61520994, -1.        ]),
     1: array([-0.21840965, -0.35501755]),
     2: array([0.21841264, 0.35502078]),
     3: array([0.61520696, 0.99999677])}

    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)

    References
    ----------
    .. [1] Fruchterman, Thomas MJ, and Edward M. Reingold.
           "Graph drawing by force-directed placement."
           Software: Practice and experience 21, no. 11 (1991): 1129-1164.
           http://dx.doi.org/10.1002/spe.4380211102
    .. [2] Hamaguchi, Hiroki, Naoki Marumo, and Akiko Takeda.
           "Initial Placement for Fruchterman--Reingold Force Model With Coordinate Newton Direction."
           arXiv preprint arXiv:2412.20317 (2024).
           https://arxiv.org/abs/2412.20317
    """
    import numpy as np

    method = "energy"  # Only energy method is supported in spring_layered_layout

    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        pos = {nx.utils.arbitrary_element(G.nodes()): center}
        if store_pos_as is not None:
            nx.set_node_attributes(G, pos, store_pos_as)
        return pos

    A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
    if k is None and fixed is not None:
        # We must adjust k by domain size for layouts not near 1x1
        nnodes, _ = A.shape
        k = dom_size / np.sqrt(nnodes)
    pos = _sparse_layered_fruchterman_reingold(
        A, k, pos_arr, fixed, iterations, threshold, dim, seed, method, gravity, nodes_layers=nodes_layers, layers_weight=layers_weight
    )

    if fixed is None and scale is not None:
        pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))

    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos


fruchterman_reingold_layered_layout = spring_layered_layout


def layered_layout_bands(nodes_layers):
    _, layers_counts = np.unique(nodes_layers, return_counts=True)

    # vols = np.log(np.e-1+layers_counts) * layers_counts
    # vols = np.pow(layers_counts, -0.5) * layers_counts
    vols = layers_counts
    vols_cumsum = np.cumsum(vols)
    radii = np.pow(vols_cumsum / np.pi, 0.5)
    # radii[0] = 0.01
    radii = np.append(radii[::-1], [0])[::-1]
    max_layer = np.max(nodes_layers)
    radii = np.arange(max_layer + 1)
    return np.stack([radii, radii + 0.5]).T


def _sparse_layered_fruchterman_reingold(
    A,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    dim=2,
    seed=None,
    method="energy",
    gravity=1.0,
    nodes_layers=None,
    layers_weight=10.0,
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    import numpy as np
    import scipy as sp

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)

    if nodes_layers is None:
        raise ValueError("Nodes layers are requred for layered FP layout")
    bands = layered_layout_bands(nodes_layers)
    bands = bands[nodes_layers]

    return _energy_layered_fruchterman_reingold(
        A,
        nnodes,
        k,
        pos,
        fixed,
        iterations,
        threshold,
        dim,
        gravity,
        bands,
        layers_weight,
    )


def _energy_layered_fruchterman_reingold(A, nnodes, k, pos, fixed, iterations, threshold, dim, gravity, bands, layers_weight):
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # energy-based version
    import numpy as np
    import scipy as sp
    print("len(fixed)=",len(fixed))
    if gravity <= 0:
        raise ValueError(f"the gravity must be positive.")

    # make sure we have a Compressed Sparse Row format
    try:
        A = A.tocsr()
    except AttributeError:
        A = sp.sparse.csr_array(A)

    # Take absolute values of edge weights and symmetrize it
    A = np.abs(A)
    A = (A + A.T) / 2

    n_components, labels = sp.sparse.csgraph.connected_components(A, directed=False)
    bincount = np.bincount(labels)
    batchsize = 500

    def _cost_FR(x):
        pos = x.reshape((nnodes, dim))
        grad = np.zeros((nnodes, dim))
        cost = 0.0
        for l in range(0, nnodes, batchsize):
            r = min(l + batchsize, nnodes)
            # difference between selected node positions and all others
            delta = pos[l:r, np.newaxis, :] - pos[np.newaxis, :, :]
            # distance between points with a minimum distance of 1e-5
            distance2 = np.sum(delta * delta, axis=2)
            distance2 = np.maximum(distance2, 1e-10)
            distance = np.sqrt(distance2)
            # temporary variable for calculation
            Ad = A[l:r] * distance
            # attractive forces and repulsive forces
            grad[l:r] = 2 * np.einsum("ij,ijk->ik", Ad / k - k**2 / distance2, delta)
            # integrated attractive forces
            cost += np.sum(Ad * distance2) / (3 * k)
            # integrated repulsive forces
            cost -= k**2 * np.sum(np.log(distance))
        # gravitational force from the centroids of connected components to (0.5, ..., 0.5)^T
        centers = np.zeros((n_components, dim))
        np.add.at(centers, labels, pos)
        delta0 = centers / bincount[:, np.newaxis] - 0.5
        grad += gravity * delta0[labels]
        cost += gravity * 0.5 * np.sum(bincount * np.linalg.norm(delta0, axis=1) ** 2)

        # additional term to move the layers to the corresponding bands
        pos_radii = np.sqrt(np.sum(np.pow(pos, 2), axis=1)).reshape(-1, 1)

        delta_lower = pos_radii - bands[:, 0:1]
        delta_higher = pos_radii - bands[:, 1:]
        mask_lower = delta_lower < 0
        mask_higher = delta_higher > 0
        delta_lower = delta_lower * mask_lower
        delta_higher = delta_higher * mask_higher

        cost_layers = np.pow(delta_lower, 2) + np.pow(delta_higher, 2)
        grad_layers = (2 * delta_lower + 2 * delta_higher) * pos / pos_radii

        cost += layers_weight * cost_layers.sum()
        grad += layers_weight * grad_layers

        # fix positions of fixed nodes
        grad[fixed] = 0.0
        return cost, grad.ravel()

    # Optimization of the energy function by L-BFGS algorithm
    options = {"maxiter": iterations, "gtol": threshold}
    return sp.optimize.minimize(_cost_FR, pos.ravel(), method="L-BFGS-B", jac=True, options=options).x.reshape((nnodes, dim))


def get_subgraph_with_fist_m_layers(nx_graph, m):
    return nx_graph.subgraph([node[0] for node in nx_graph.nodes(data=True) if node[1]["layer_"] <= m])


def layered_sequential_FD_layout(
    nx_graph,
    k_modifiers_no_layers=1.0,
    k_modifiers_layers=1.0,
    layers_weights=200.0,
):

    nodes_layers = np.array([x[1]["layer_"] for x in nx_graph.nodes(data=True)])
    max_diameter = max(nodes_layers)
    subgraph = get_subgraph_with_fist_m_layers(nx_graph, 0)
    pos = {k: (0.0, 0.0) for k in subgraph.nodes}
    fixed = list(pos.keys())

    if not isinstance(k_modifiers_no_layers, list):
        k_modifiers_no_layers = [k_modifiers_no_layers] * max_diameter
    if not isinstance(k_modifiers_layers, list):
        k_modifiers_layers = [k_modifiers_layers] * max_diameter

    if not isinstance(layers_weights, list):
        layers_weights = [layers_weights] * max_diameter
    # nx.set_node_attributes(nx_graph, pos, "pos_")
    assert len(k_modifiers_no_layers) == max_diameter, "number of k_modifiers_no_layers must be the same as the number of layers"
    assert len(k_modifiers_layers) == max_diameter, "number of k_modifiers_layers must be the same as the number of layers"
    assert len(layers_weights) == max_diameter, "number of layers_weights must be the same as the number of layers"
    for max_layer, k1, k2, lw in zip(range(1, max_diameter + 1), k_modifiers_no_layers, k_modifiers_layers, layers_weights):
        # nx_graph = bfs_result_to_nx_graph(bfs_result, max_layer=max_layer)
        subgraph = get_subgraph_with_fist_m_layers(nx_graph, max_layer)
        nodes_layers = np.array([x[1]["layer_"] for x in subgraph.nodes(data=True)])
        k_scaling = 1 / np.sqrt(len(subgraph.nodes))
        pos = fruchterman_reingold_layout(subgraph, pos=pos, fixed=fixed, k=k1 * k_scaling)
        pos = fruchterman_reingold_layered_layout(subgraph, pos=pos, fixed=fixed, nodes_layers=nodes_layers, layers_weight=lw, k=k2 * k_scaling)
        fixed = list(pos.keys())
    nx.set_node_attributes(nx_graph, pos, "pos_")
    return pos


# def draw_graph_with_igraph(
#         nx_graph,
#         cmap = colormaps["Set1"],
#         layout: Literal["kamada_kawai", "fruchterman_reingold"] = "kamada_kawai",
#         node_alpha = 1., # not used
#         node_size = 8,
#         node_linewidth = 1., # not used
#         edge_alpha=0.2,
#         edge_width=0.4,
#         figsize = (10,10),
#         fig=None,
#         ax=None,
#     ):
#     ig = IGraph.from_networkx(nx_graph)
#     layers_ids = [nx_graph.nodes[node]['layer_'] for node in nx_graph.nodes]
#     ig.vs['color'] = [colors.to_hex(cmap(x)) for x in layers_ids]
#     alpha_hex = f'{int(edge_alpha * 255):02x}'
#     ig.es['color'] = [f'#000000{alpha_hex}'] * ig.ecount()  #

#     fig, ax = get_fig_ax(fig, ax, figsize=figsize)

#     igraph.plot(ig, target=ax, vertex_size=node_size, edge_width=edge_width, layout=getattr(ig, f"layout_{layout}")())
#     legend_elements = [Patch(facecolor=cmap(x), label=x) for x in set(layers_ids)]
#     ax.legend(handles=legend_elements, title='Layer')
#     return fig, ax
