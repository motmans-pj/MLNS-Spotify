import numpy as np
import csv
import networkx as ntx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

###################
# Utils for link prediction 
###################


def read_graph_file(
        filename: str
    )-> list:
    """
    read_graph_file(filename):
        reads a txt file containing the graph data and returns a list of lists
    (just made a function out of the public_baseline.py professor's code)
    TESTED (SEEMS TO DO ITS JOB DECENTLY)
    """
    with open(filename, "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)
    test_set = [element[0].split(" ") for element in test_set]
    return test_set

def write_link_predictions(
        predictions: zip,
        submission_file: str ,#| Path, 
        id_field: str = "ID", 
        prediction_field: str = "Predicted"
    )-> None:
    """
    write_link_prediction(pred, submission_file, id_field, prediction_field):
        write the predictions to disk
    (just made a function out of the public_baseline.py professor's code)
    NOT TESTED, IT MAY  NOT WORK
    """
    with open(submission_file,"w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(i for i in [id_field, prediction_field])
        for row in predictions:
            csv_out.writerow(row)
        pred.close()

def init_netx_graph(
        graph_list,#: list[list],
        train: bool = True
    )-> ntx.DiGraph:
    """
    init_netx_graph(graph_list, train):
        initialize the Networkx DiGraph object from a list,
        If train, it initializes also the edges, otherwise just the nodes
    TESTED (SEEMS TO DO ITS JOB DECENTLY)
    """
    graph = ntx.DiGraph()
    if train:
        for source, target, edge in graph_list:
            source = int(source); target = int(target); edge = bool(edge)
            graph.add_node(source)
            graph.add_node(target)
            if edge: 
                graph.add_edge(source, target)
    else:
        for source, target in graph_list:
            source = int(source); target = int(target)
            graph.add_node(source)
            graph.add_node(target)
    return graph

def plot_graph(
        graph: ntx.DiGraph
    )-> None:
    """
    plot_graph(graph):
        plot ntx.DiGraph with matplotlli.pyplot
    the graph is still ugly, probably this is useless
    TESTED (SEEMS TO DO ITS JOB DECENTLY)
    """
    pos = ntx.spring_layout(graph)
    ntx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_size = 1)
    ntx.draw_networkx_edges(graph, pos, edge_color='r', arrows=True)
    ntx.draw_networkx_edges(graph, pos,  arrows=False)
    plt.show()

def split_validation_graph(
        graph: ntx.DiGraph,
        random_state: int,
        val_size: float,
        split_networks: bool = False
    ):#-> tuple[ntx.DiGraph]:
    """
    split_validation_graph(graph, random_state, val_size):
        splits training graph in train and validation set given random state and validation size
        keeping the same set of node for both training and validation as we want to predict the 
        creation of links bewteen existing nodes and not whether new nodes will disappear.
        If split_networks is True, it will isolate the nodes of the two graphs 
        (I don't know in which way it is correct, i'd keep it False by default)
    TESTED (SEEMS TO DO ITS JOB DECENTLY)
    """
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    edges_train, edges_test = train_test_split(edges, test_size = val_size, random_state = random_state)
    nodes_train, nodes_test = train_test_split(nodes, test_size = val_size, random_state = random_state)
    G_train = ntx.DiGraph()
    G_test = ntx.DiGraph()   
    if not split_networks:
        G_train.add_nodes_from(graph.nodes())
        G_test.add_nodes_from(graph.nodes())
    else:
        G_train.add_nodes_from(nodes_train)
        G_test.add_edge(nodes_test)
    G_train.add_edges_from(edges_train)
    G_test.add_edges_from(edges_test)
    return G_train, G_test