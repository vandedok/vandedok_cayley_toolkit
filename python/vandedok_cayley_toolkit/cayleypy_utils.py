import networkx as nx


def get_vertex_names_with_layers(bfs_result, max_layer=None) -> list[str]:
    """Returns names for vertices in the graph."""
    names = []
    layers = []
    delimiter = "" if int(max(bfs_result.graph.central_state)) <= 9 else ","

    for layer_id in range(min(len(bfs_result.layers), max_layer + 1)):
        if layer_id not in bfs_result.layers:
            raise ValueError("To get explicit graph, run bfs with max_layer_size_to_store=None.")

        for state in bfs_result.get_layer(layer_id):
            names.append(delimiter.join(str(int(x)) for x in state))
            layers.append(layer_id)
    return names, layers


def bfs_result_to_nx_graph(bfs_result, max_layer=None):
    vertex_names, layers_ids = get_vertex_names_with_layers(bfs_result, max_layer=max_layer)
    num_vertices = len(vertex_names)
    nx_graph = nx.Graph()
    for name, layer_id in zip(vertex_names, layers_ids):
        nx_graph.add_node(name, layer_=layer_id)
    for i1, i2 in bfs_result.edges_list:
        label = bfs_result.get_edge_name(i1, i2)
        if i1 < num_vertices and i2 < num_vertices:
            nx_graph.add_edge(vertex_names[i1], vertex_names[i2], label=label)
    return nx_graph
