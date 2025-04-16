# Carregando Bibliotegas e Funções
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import json
from geopy.distance import great_circle
from grakel import graph_from_networkx, GraphKernel
import networkx as nx
import pickle
import duckdb
from math import radians, sin, cos, sqrt, atan2
from fastdtw import fastdtw


# Função para calcular o número mínimo de operações (adição, remoção, substituição de nós ou arestas)
def distancia_edit(g1, g2):
    ged = nx.graph_edit_distance(g1, g2)
    return ged


# Função para calcular o similaridade Jaccard entre os conjuntos de arestas
def jaccard_similarity(g1, g2):
    edges_G1 = set(g1.edges())
    edges_G2 = set(g2.edges())
    return len(edges_G1.intersection(edges_G2)) / len(edges_G1.union(edges_G2))


# Graph kernels são técnicas mais avançadas para medir similaridade entre grafos, baseadas em transformações matemáticas dos grafos em vetores

def graph_kernel(g1, g2):
    # Converter grafos NetworkX para formato utilizado pelo GraKeL
    G1_grakel = graph_from_networkx([g1], node_labels_tag=None, edge_labels_tag=None)
    G2_grakel = graph_from_networkx([g2], node_labels_tag=None, edge_labels_tag=None)
    # Escolher um kernel, por exemplo, o Weisfeiler-Lehman kernel
    gk = GraphKernel(kernel=["weisfeiler_lehman", "subtree_wl"])
    # Calcular similaridade
    return gk.fit_transform(G1_grakel + G2_grakel)

# abordagem mede a similaridade encontrando subgrafos correspondentes entre dois grafos. NetworkX tem ferramentas para subgraph isomorphism
def subgraph_matching(g1, g2):
    return nx.algorithms.isomorphism.GraphMatcher(g1, g2).subgraph_is_isomorphic()

def cne_similarity(rota1, rota2):
    nodes1 = set(rota1.nos)
    nodes2 = set(rota2.nos)
    common_nodes = nodes1 & nodes2
    max_nodes = max(len(nodes1), len(nodes2))
    total_similarity = len(common_nodes)
    return total_similarity / max_nodes if max_nodes > 0 else 1

def similaridade_grafos(g1, g2):
    pos_g1 = nx.get_node_attributes(g1, "pos")
    pos_g2 = nx.get_node_attributes(g2, "pos")
    ponto = 0
    for node1 in pos_g1:
        coords_g1 = pos_g1[node1]
        existe_ponto = False
        for node2 in pos_g2:
            coords_g2 = pos_g2[node2]
            km = great_circle(coords_g1, coords_g2).kilometers
            if km <= 0.2:
                existe_ponto = True
                break
        if existe_ponto:
            ponto = ponto + 1
    return ponto / len(pos_g1)

class DTWDistance:
    def _calc(self, x, y):
        """
        Calcula a distância de Haversine entre dois pontos (lat, lon).
        """
        R = 6371.0  # Raio da Terra em km

        lat1, lon1 = map(radians, x)
        lat2, lon2 = map(radians, y)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c  # Retorna a distância em km

    def calculate(self, route1, route2):
        """
        Calcula a distância DTW entre duas rotas de pontos geográficos.

        :param route1: np.array com shape (N, 2) - Primeira rota (lat, lon)
        :param route2: np.array com shape (M, 2) - Segunda rota (lat, lon)
        :return: Distância DTW entre as rotas
        """
        distance, path = fastdtw(route1, route2, dist=self._calc)
        return distance

class DiscreteFrechetDistance:
    def calculate(self, curve1, curve2):
        n, m = len(curve1), len(curve2)
        dp = np.zeros((n, m))
        dp[0, 0] = np.linalg.norm(curve1[0] - curve2[0])
        for i in range(1, n):
            dp[i, 0] = max(dp[i - 1, 0], np.linalg.norm(curve1[i] - curve2[0]))
        for j in range(1, m):
            dp[0, j] = max(dp[0, j - 1], np.linalg.norm(curve1[0] - curve2[j]))
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(
                    min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                    np.linalg.norm(curve1[i] - curve2[j]),
                )
        return dp[-1, -1]


def matriz_similaridade(lista_grafos, medida_de_distancia="similarity"):
    g = list(lista_grafos.items())
    n = len(lista_grafos)
    similaridade_matrix = np.zeros((n, n))
    # Calcula apenas metade superior da matriz (já que é simétrica)
    for l1 in range(n):
        for l2 in range(l1, n):
            if medida_de_distancia == "jaccard":
                sim = jaccard_similarity(g[l1][1], g[l2][1])
            elif medida_de_distancia == "graph_kernel":
                sim = graph_kernel(g[l1][1], g[l2][1])
            elif medida_de_distancia == "subgraph_matching":
                sim = subgraph_matching(g[l1][1], g[l2][1])
            elif medida_de_distancia == "distancia_edit":
                sim = distancia_edit(g[l1][1], g[l2][1])
            elif medida_de_distancia == "similarity":
                if l1 == l2:
                    sim = 1
                else:
                    sim = (
                        similaridade_grafos(g[l1][1], g[l2][1])
                        + similaridade_grafos(g[l2][1], g[l1][1])
                    ) / 2
            similaridade_matrix[l1, l2] = sim
            similaridade_matrix[l2, l1] = sim  # Matriz simétrica
    return similaridade_matrix
