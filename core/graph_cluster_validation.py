import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import pairwise_distances


# Função para calcular a Silhueta
def silhouette_clustering(X, labels):
    return silhouette_score(X, labels)


# Função para calcular o Davies-Bouldin index
def davies_bouldin_clustering(X, labels):
    return davies_bouldin_score(X, labels)


# Função para calcular a Distância Média Inter-Cluster
def mean_inter_cluster_distance(X, labels):
    unique_labels = np.unique(labels)
    distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            dist = np.mean(pairwise_distances(cluster_i, cluster_j))
            distances.append(dist)
    return np.mean(distances)


# Função para calcular a Modularidade
def modularity(graph, labels):
    return nx.community.modularity(graph, [np.where(labels == i)[0] for i in np.unique(labels)])


# Função para calcular a Conductance
def conductance(graph, labels):
    clusters = np.unique(labels)
    conductances = []
    for cluster in clusters:
        # Indices dos nós no cluster
        cluster_nodes = np.where(labels == cluster)[0]
        # Vizinhos fora do cluster
        external_nodes = np.where(labels != cluster)[0]
        # Arestas dentro do cluster
        internal_edges = sum(1 for i in cluster_nodes for j in cluster_nodes if graph.has_edge(i, j))
        # Arestas para o exterior do cluster
        external_edges = sum(1 for i in cluster_nodes for j in external_nodes if graph.has_edge(i, j))
        conductance_value = external_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
        conductances.append(conductance_value)
    return np.mean(conductances)


# Função para calcular o Normalized Cut (NCut)
def normalized_cut(graph, labels):
    clusters = np.unique(labels)
    ncut_values = []
    for cluster in clusters:
        cluster_nodes = np.where(labels == cluster)[0]
        external_nodes = np.where(labels != cluster)[0]
        # Arestas dentro do cluster
        internal_edges = sum(1 for i in cluster_nodes for j in cluster_nodes if graph.has_edge(i, j))
        # Arestas para o exterior do cluster
        external_edges = sum(1 for i in cluster_nodes for j in external_nodes if graph.has_edge(i, j))
        ncut_value = external_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
        ncut_values.append(ncut_value)
    return np.mean(ncut_values)


# Função para calcular o Rand Index (RI)
def rand_index(labels_true, labels_pred):
    from sklearn.metrics import rand_score
    return rand_score(labels_true, labels_pred)


# Função para calcular o Adjusted Rand Index (ARI)
def adjusted_rand_index(labels_true, labels_pred):

    return adjusted_rand_score(labels_true, labels_pred)


# Função para calcular a Normalized Mutual Information (NMI)
def normalized_mutual_information(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


# Função principal para calcular todas as medidas de validação de agrupamentos
def clustering_validation_measures(X, labels_true, labels_pred, graph=None):
    # Calculando as medidas
    silhouette = silhouette_clustering(X, labels_pred)
    davies_bouldin = davies_bouldin_clustering(X, labels_pred)
    mean_inter_cluster = mean_inter_cluster_distance(X, labels_pred)
    # Modularity e Conductance requerem um grafo
    if graph is not None:
        modularity_value = modularity(graph, labels_pred)
        conductance_value = conductance(graph, labels_pred)
        ncut_value = normalized_cut(graph, labels_pred)
    else:
        modularity_value = None
        conductance_value = None
        ncut_value = None
    rand_index_value = rand_index(labels_true, labels_pred)
    ari_value = adjusted_rand_index(labels_true, labels_pred)
    nmi_value = normalized_mutual_information(labels_true, labels_pred)
    # Retornar todos os valores calculados
    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin,
        "Mean Inter-Cluster Distance": mean_inter_cluster,
        "Modularity": modularity_value,
        "Conductance": conductance_value,
        "Normalized Cut": ncut_value,
        "Rand Index": rand_index_value,
        "Adjusted Rand Index": ari_value,
        "Normalized Mutual Information": nmi_value
    }