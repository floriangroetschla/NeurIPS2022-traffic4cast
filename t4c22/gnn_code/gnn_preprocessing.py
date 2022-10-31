import math
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

def get_compacted_graph(data):
    # Build graph
    data.id = torch.tensor([i for i in range(data.x.shape[0])])
    G = to_networkx(data, node_attrs=['id', 'x'], edge_attrs=['edge_attr', 'y'], to_undirected=False, remove_self_loops=False)#.to_undirected()
    nx.set_edge_attributes(G, {(u, v): {'parsed_maxspeed': G.edges[u,v]['edge_attr'][0], 'importance': G.edges[u,v]['edge_attr'][1], 'length_meters': G.edges[u,v]['edge_attr'][2]} for u, v, ed in G.edges.data()})

    nx.set_edge_attributes(G, {(u, v): {'orig_edges': [(u,v)]} for u, v, ed in G.edges.data()})

    super_segment_nodes = set()
    
    for i in range(data.edge_index_supergraph.shape[1]):
        super_segment_nodes.add(data.edge_index_supergraph[0][i].item())
        super_segment_nodes.add(data.edge_index_supergraph[1][i].item())
                        
    # Contract edges
    for node in list(G.nodes):
        if node in super_segment_nodes:
            pass
        elif G.degree[node] == 0:
            G.remove_node(node)
        elif G.in_degree[node] == 2 and G.out_degree[node] == 2 and (sorted(list(G.predecessors(node))) == sorted(list(G.neighbors(node)))):
            in_neighbor = list(G.predecessors(node))[0]
            out_neighbor = list(G.predecessors(node))[1]
            if not (G.has_edge(in_neighbor, out_neighbor) or G.has_edge(out_neighbor, in_neighbor)):
                replaced_edges_in_out = G.edges[in_neighbor, node]['orig_edges'] + G.edges[node, out_neighbor]['orig_edges']
                replaced_edges_out_in = G.edges[out_neighbor, node]['orig_edges'] + G.edges[node, in_neighbor]['orig_edges']
                parsed_maxspeed_in_out = G.edges[in_neighbor, node]['parsed_maxspeed'] + G.edges[node, out_neighbor]['parsed_maxspeed'] / 2
                parsed_maxspeed_out_in = G.edges[out_neighbor, node]['parsed_maxspeed'] + G.edges[node, in_neighbor]['parsed_maxspeed'] / 2
                importance_in_out = G.edges[in_neighbor, node]['importance'] + G.edges[node, out_neighbor]['importance'] / 2
                importance_out_in = G.edges[out_neighbor, node]['importance'] + G.edges[node, in_neighbor]['importance'] / 2
                length_meters_in_out = G.edges[in_neighbor, node]['length_meters'] + G.edges[node, out_neighbor]['length_meters']
                length_meters_out_in = G.edges[out_neighbor, node]['length_meters'] + G.edges[node, in_neighbor]['length_meters']
                y_in_out = G.edges[in_neighbor, node]['y']
                y_out_in = G.edges[out_neighbor, node]['y']
                old_attributes = G.nodes[node]
                G.remove_node(node)
                G.add_edge(in_neighbor, out_neighbor, orig_edges=replaced_edges_in_out, parsed_maxspeed=parsed_maxspeed_in_out, importance=importance_in_out, length_meters=length_meters_in_out, y=y_in_out, edge_attr=[0, 0, 0])
                G.add_edge(out_neighbor, in_neighbor, orig_edges=replaced_edges_out_in, parsed_maxspeed=parsed_maxspeed_out_in, importance=importance_out_in, length_meters=length_meters_out_in, y=y_out_in, edge_attr=[0, 0, 0])
                for neighbor in [in_neighbor, out_neighbor]:
                    for i in range(data.x.shape[1]):
                        if math.isnan(G.nodes[neighbor]['x'][i]):
                            G.nodes[neighbor]['x'][i] = old_attributes['x'][i]
                
                                    
    # Build new data object
    data_c = from_networkx(G, group_node_attrs=['id', 'x'], group_edge_attrs=['parsed_maxspeed', 'importance', 'length_meters'])
    
    # Create node mapping
    node_mapping = [-1 for i in range(data.x.shape[0])]    # from original to compressed
    for i in range(data_c.x.shape[0]):
        node_mapping[int(data_c.x[i][0].item())] = i
    
    # Edge mapping
    edge_mapping = {}    # from original to compressed
    for edge in G.edges:
        for orig_edge in G.edges[edge]['orig_edges']:
            edge_mapping[orig_edge] = (node_mapping[edge[0]], node_mapping[edge[1]])
            
    edge_map = {}
    for i in range(data_c.edge_index.shape[1]):
        edge_map[(data_c.edge_index[0][i].item(), data_c.edge_index[1][i].item())] = i
                                    
    # Create edge index index
    edge_index_index = []   # from original to compressed
    for i in range(data.edge_index.shape[1]):
        u = data.edge_index[0][i].item()
        v = data.edge_index[1][i].item()
        if (u, v) in edge_mapping:
            u_c, v_c = edge_mapping[(u,v)]
        else:
            v_c, u_c = edge_mapping[(v,u)]
        edge_index_index.append(edge_map[(u_c, v_c)])
        
    data_c.x = data_c.x[:,1:]  # remove id from output
    del data_c.orig_edges

    data_c.y_orig = data.y
    data_c.edge_index_index = torch.tensor(edge_index_index)  # edge_index_index maps from data.edge_index to data_c.edge_index
    data_c.x_orig = data.x
    data_c.edge_attr_orig = data.edge_attr
    data_c.edge_index_orig = data.edge_index  
    data_c.edge_index_supergraph = data.edge_index_supergraph
    data_c.node_mapping = torch.tensor(node_mapping)
    data_c.y_eta = data.y_eta

    for i in range(data_c.edge_index_supergraph.shape[1]):
        data_c.edge_index_supergraph[0][i] = node_mapping[int(data_c.edge_index_supergraph[0][i].item())]
        data_c.edge_index_supergraph[1][i] = node_mapping[int(data_c.edge_index_supergraph[1][i].item())]

    return data_c   # edge_index_index maps from data.edge_index to data_c.edge_index
