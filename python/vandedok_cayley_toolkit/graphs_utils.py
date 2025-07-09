import networkx as nx
import numpy as np
from networkx.drawing.layout import (
    random_layout,
    circular_layout,
    rescale_layout,
    _kamada_kawai_solve,
    _process_params,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
from .matplotlib_utils import get_fig_ax, lighten


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

    # Additianal term to move the layers to the corresponding bands

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
