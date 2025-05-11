# Carregando Bibliotegas e Funções
import numpy as np
from sklearn.metrics import pairwise_distances
from geopy.distance import great_circle
from grakel import graph_from_networkx, GraphKernel
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from fastdtw import fastdtw
from networkx.algorithms import isomorphism
from networkx.algorithms.similarity import optimize_graph_edit_distance
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize, binarize
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine, cityblock, jaccard
from tslearn.metrics import dtw as ts_dtw
import Levenshtein as lev
from geopy.distance import geodesic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


# Cosine Similarity
def cosine_similarity(g1, g2):
    # Transforma os nós dos grafos em "documentos" de texto
    doc1 = ' '.join(str(node) for node in g1.nodes)
    doc2 = ' '.join(str(node) for node in g2.nodes)

    # Vetorização com presença de nós
    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform([doc1, doc2]).toarray()

    # Similaridade cosseno entre os vetores
    sim = sklearn_cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return sim


# Distância Geodésica
def geodesic_distance(g1, g2):
    # Calcula a distância geodésica entre os grafos considerando os menores caminhos entre os nós
    def centroide(g):
        latitudes = [g.nodes[coord]['lat'] for coord in g.nodes]
        longitudes = [g.nodes[coord]['lon'] for coord in g.nodes]
        return (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))
    
    centroide1 = centroide(g1)
    centroide2 = centroide(g2)
    return great_circle(centroide1, centroide2).kilometers


# Distância ponderada
def weighted_distance(g1, g2):
    dist_total = 0
    pares_comparados = 0

    for node1 in g1.nodes():
        for node2 in g1.nodes():
            if node1 == node2:
                continue

            try:
                path_length_g1 = nx.shortest_path_length(g1, source=node1, target=node2, weight='weight')
            except nx.NetworkXNoPath:
                continue

            try:
                coord1 = (g1.nodes[node1]['lat'], g1.nodes[node1]['lon'])
                coord2 = (g1.nodes[node2]['lat'], g1.nodes[node2]['lon'])
            except KeyError:
                continue

            try:
                closest_node1 = min(g2.nodes, key=lambda x: great_circle(coord1, (g2.nodes[x]['lat'], g2.nodes[x]['lon'])).meters)
                closest_node2 = min(g2.nodes, key=lambda x: great_circle(coord2, (g2.nodes[x]['lat'], g2.nodes[x]['lon'])).meters)
            except (KeyError, ValueError):
                continue

            try:
                path_length_g2 = nx.shortest_path_length(g2, source=closest_node1, target=closest_node2, weight='weight')
            except nx.NetworkXNoPath:
                continue

            dist_total += abs(path_length_g1 - path_length_g2)
            pares_comparados += 1

    if pares_comparados == 0:
        return float('inf')
    
    return dist_total / pares_comparados



# Distância de Caminho (Path Distance)
def path_distance(g1, g2):
   # Verifica nós em comum
    common_nodes = set(g1.nodes()).intersection(set(g2.nodes()))
    
    if not common_nodes:
        return float('inf')  # Nenhum nó em comum, distância infinita

    dist = 0
    for node in common_nodes:
        lengths_g1 = nx.single_source_shortest_path_length(g1, node)
        lengths_g2 = nx.single_source_shortest_path_length(g2, node)
        
        # Soma diferenças apenas nos nós também presentes no outro grafo
        shared_targets = set(lengths_g1).intersection(lengths_g2)
        for target in shared_targets:
            dist += abs(lengths_g1[target] - lengths_g2[target])
    
    return dist


# Função para calcular o número mínimo de operações (adição, remoção, substituição de nós ou arestas)
def distancia_edit(g1, g2):
    """
    Calcula a distância de edição entre dois grafos.
    """
    # Converter arestas para conjuntos para facilitar a comparação
    edges_g1 = set(g1.edges())
    edges_g2 = set(g2.edges())

    # Operações de edição
    arestas_removidas = edges_g1 - edges_g2  # Arestas em g1 que não estão em g2
    arestas_adicionadas = edges_g2 - edges_g1  # Arestas em g2 que não estão em g1

    # Distância de edição é a soma das operações
    distancia = len(arestas_removidas) + len(arestas_adicionadas)

    return distancia


# Função para calcular o similaridade Jaccard entre os conjuntos de arestas
def jaccard_similarity(g1, g2):
    """
    Calcula a similaridade de Jaccard entre dois grafos.
    """
    edges_g1 = set(g1.edges())
    edges_g2 = set(g2.edges())

    # Calcular a interseção e a união dos conjuntos de arestas
    intersecao = len(edges_g1.intersection(edges_g2))
    uniao = len(edges_g1.union(edges_g2))

    # Evitar divisão por zero
    if uniao == 0:
        return 0.0

    # Similaridade de Jaccard
    return intersecao / uniao


# Graph kernels são técnicas mais avançadas para medir similaridade entre grafos, baseadas em transformações matemáticas dos grafos em vetores

def graph_kernel(g1, g2, label="label"):
    """
    Calcula a similaridade entre dois grafos usando Graph Kernels.
    """
    # Converter grafos NetworkX para formato utilizado pelo GraKeL
    G1_grakel = list(graph_from_networkx([g1], node_labels_tag=label, edge_labels_tag=None))
    G2_grakel = list(graph_from_networkx([g2], node_labels_tag=label, edge_labels_tag=None))
    # Escolher um kernel, por exemplo, o Weisfeiler-Lehman kernel
    gk = GraphKernel(kernel=["weisfeiler_lehman", "subtree_wl"])
    # Calcular similaridade
    sim_matrix = gk.fit_transform([G1_grakel[0], G2_grakel[0]])
    return sim_matrix[0,1] 


def similaridade_grafos(g1, g2):
    """
    Calcula a similaridade entre dois grafos com base na distância de Haversine entre os nós.
    """
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
    return ponto / len(pos_g1) if len(pos_g1) > 0 else 0


# abordagem mede a similaridade encontrando subgrafos correspondentes entre dois grafos. NetworkX tem ferramentas para subgraph isomorphism
def subgraph_matching(g1, g2):
    """ 
    Verifica se dois grafos são isomorfos, ou seja, se têm a mesma estrutura.
    """
    return nx.algorithms.isomorphism.GraphMatcher(g1, g2).subgraph_is_isomorphic()


def cne_similarity(g1, g2):
    """ 
    Calcula a similaridade entre duas rotas com base nos nós comuns.
    """
    nodes1 = set(g1.nodes)
    nodes2 = set(g2.nodes)
    common_nodes = nodes1 & nodes2
    max_nodes = max(len(nodes1), len(nodes2))
    total_similarity = len(common_nodes)
    return total_similarity / max_nodes if max_nodes > 0 else 1


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

    def calculate(self, g1, g2):
        """
        Calcula a distância DTW entre duas rotas de pontos geográficos.

        :param route1: np.array com shape (N, 2) - Primeira rota (lat, lon)
        :param route2: np.array com shape (M, 2) - Segunda rota (lat, lon)
        :return: Distância DTW entre as rotas
        """
        distance, path = fastdtw(g1, g2, dist=self._calc)
        return distance


def safe_diameter(g):
    if not nx.is_strongly_connected(g):
        return float('inf')  # Ou algum valor default, ou pule esse par
    return nx.diameter(g)


# Earth Mover’s Distance (EMD)
def emd_distance(g1, g2):
    # Aproximação da distância de Earth Mover (EMD) usando o comprimento das distâncias
    d1 = safe_diameter(g1)
    d2 = safe_diameter(g2)
    if float('inf') in (d1, d2):
        return 0  # Similaridade nula se não for possível calcular
    return d1 - d2  # Exemplo simplificado


def diameter_similarity(g1, g2):    
    d1 = safe_diameter(g1)
    d2 = safe_diameter(g2)
    if float('inf') in (d1, d2):
        return 0  # Similaridade nula se não for possível calcular
    return 1 - abs(d1 - d2) / max(d1, d2)


def frechet_similarity(g1, g2):
    """
    Calcula a similaridade de Frechet entre duas curvas.
    """
    curve1 = extract_coord_sequence(g1)
    curve2 = extract_coord_sequence(g2)
    return frechet_distance(curve1, curve2)


def frechet_distance(curve1, curve2):
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


## Adaptar para function injection
# Informar simetria
def matriz_similaridade(lista_grafos, func_similaridade):
    """
    Calcula a matriz de similaridade entre grafos usando uma função de similaridade injetada.

    :param lista_grafos: Dicionário de grafos {nome: grafo}
    :param func_similaridade: Função que calcula a similaridade entre dois grafos
    :return: Matriz de similaridade (numpy array)
    """
    g = lista_grafos
    n = len(lista_grafos)
    similaridade_matrix = np.zeros((n, n))

    # Calcula apenas metade superior da matriz (já que é simétrica)
    for l1 in range(n):
        for l2 in range(l1, n):
            if l1 == l2:
                sim = 1  # Similaridade máxima para o mesmo grafo
            else:
                sim = func_similaridade(g[l1], g[l2])
                sim = 0.0 if np.isnan(sim) else sim  # trata NaN como 0
            similaridade_matrix[l1, l2] = sim
            similaridade_matrix[l2, l1] = sim  # Matriz simétrica

    return similaridade_matrix


def matriz_adjacencia(lista_rotas, grafo):
    n_max = len(grafo.nos)
    r_max = len(lista_rotas)
    nos_grafo = list(grafo.nos)
    matriz_adjacencia = np.zeros((r_max, n_max))
    for idx_r, rota in enumerate(lista_rotas):
        for no in rota.nos:
            if no in nos_grafo:
                idx_no = nos_grafo.index(no)
                matriz_adjacencia[idx_r][idx_no] = 1
    return matriz_adjacencia


## Normaliza Matriz
def normaliza_matriz(matriz):  
    matrix_min = np.min(matriz)
    matrix_max = np.max(matriz)
    matrix_normalized = (matriz - matrix_min) / (matrix_max - matrix_min)
    return matrix_normalized

## Revisar para Incluir no Codigo
# Autor: Lucas de Oliveira
# Data: 2023-06-21
# Função para calcular o número mínimo de operações (adição, remoção, substituição de nós ou arestas)

# Graph kernels são técnicas mais avançadas para medir similaridade entre grafos, baseadas em transformações matemáticas dos grafos em vetores


def subgraph_similarity(g1, g2):
    GM = isomorphism.GraphMatcher(g1, g2)
    # Tenta encontrar uma correspondência de subgrafos
    matches = list(GM.subgraph_isomorphisms_iter())
    if not matches:
        return 0.0  
    # Pega o melhor match
    best_match = max(matches, key=lambda m: len(m))
    match_size = len(best_match)
    # Normaliza pela quantidade de nós do menor grafo
    min_nodes = min(len(g1.nodes), len(g2.nodes))
    return match_size / min_nodes


# versao teoricamente mais rapida do subgraph
def subgraph_similarity_rapida(g1, g2):
    # Interseção e união dos nós
    nodes_g1 = set(g1.nodes())
    nodes_g2 = set(g2.nodes())
    node_intersection = len(nodes_g1 & nodes_g2)
    node_union = len(nodes_g1 | nodes_g2)
    # Interseção e união das arestas (como tuplas de pares)
    edges_g1 = set(g1.edges())
    edges_g2 = set(g2.edges())
    edge_intersection = len(edges_g1 & edges_g2)
    edge_union = len(edges_g1 | edges_g2)

    # Similaridade baseada na média das Jaccards de nós e arestas
    node_sim = node_intersection / node_union if node_union else 0
    edge_sim = edge_intersection / edge_union if edge_union else 0

    return (node_sim + edge_sim) / 2


# -------- Funções para o DTW ----------
def extract_coord_sequence(g):
    # Extrai coordenadas na ordem dos nós (baseado na sequência da rota)
    coords = [g.nodes[n]['pos'] for n in g.nodes()]
    return np.array(coords)


def dtw_similarity(g1, g2):
    seq1 = extract_coord_sequence(g1)
    seq2 = extract_coord_sequence(g2)

    return ts_dtw(seq1, seq2)


# -------- Funções para o LCS ----------
def extract_coord_sequence_str(g):
    # Converte coordenadas em strings discretas (ex: "12.345,-38.456")
    return [f"{round(lat, 4)},{round(lon, 4)}" for lat, lon in extract_coord_sequence(g)]


def lcs(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def lcs_similarity(g1, g2):
    s1 = extract_coord_sequence_str(g1)
    s2 = extract_coord_sequence_str(g2)

    lcs_len = lcs(s1, s2)
    max_len = max(len(s1), len(s2))
    return (lcs_len / max_len)  


# -------- Funções para o Levenshtein ----------
def levenshtein_similarity(g1, g2):
    seq1 = extract_coord_sequence_str(g1)
    seq2 = extract_coord_sequence_str(g2)

    # Transforma a sequência de coordenadas em strings únicas
    s1 = '|'.join(seq1)
    s2 = '|'.join(seq2)

    dist = lev.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return dist / max_len if max_len > 0 else 0


# Função principal para calcular todas as distâncias entre dois grafos
def calculate_all_distances(graph1, graph2):
    distances = {
        'Geodesic Distance': geodesic_distance(graph1, graph2),
        'Weighted Distance': weighted_distance(graph1, graph2),
        # 'Graph Diameter': diameter_similarity(graph1, graph2),
        'Frechet Distance': frechet_similarity(graph1, graph2),
        'DTW Distance': dtw_similarity(graph1, graph2),
        # 'Subgraph Similarity': subgraph_similarity(graph1, graph2),
        'LCS Distance': lcs_similarity(graph1, graph2),
        'Graph Edit Distance': distancia_edit(graph1, graph2),
        'Levenshtein Distance': levenshtein_similarity(graph1, graph2),
        'Cosine Similarity': cosine_similarity(graph1, graph2),
        'Jaccard Similarity': jaccard_similarity(graph1, graph2),
        'EMD Distance': emd_distance(graph1, graph2),
        'Path Distance': path_distance(graph1, graph2),
        'Graph Kernel': graph_kernel(graph1, graph2),
        'Similarity Haversine': similaridade_grafos(graph1, graph2),
        'CNE Similarity': cne_similarity(graph1, graph2),
    }
    return distances
