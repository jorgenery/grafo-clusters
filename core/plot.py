import logging
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.manifold import MDS
import mpld3
from mpld3 import plugins
import pydeck as pdk
import hashlib
import random
from matplotlib.patches import Rectangle
from collections import defaultdict, Counter, OrderedDict
import core.measures as measures
import skfuzzy
from sklearn.metrics import silhouette_score


def plot_matriz_distancias(distancias):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        distancias,
        annot=True,  # Mostrar os valores dentro dos quadrados
        fmt=".2f",  # Formato dos valores (2 casas decimais)
        cmap="YlGnBu",  # Mapa de cores (pode ser alterado)
        vmin=0,  # Valor mínimo da escala
        vmax=1,  # Valor máximo da escala
        linewidths=0.5,  # Espessura das linhas entre as células
        square=True,  # Manter as células quadradas
    )

    # Adicionar título e rótulos
    plt.title("Matriz de Similaridade", fontsize=16)
    plt.xlabel("Índices", fontsize=12)
    plt.ylabel("Índices", fontsize=12)
    return plt


def plot_graf_similaridade(similarity_matrix, threshold=0.5):
    # Criar um grafo a partir da matriz de similaridade
    G = nx.Graph()
    n = similarity_matrix.shape[0]

    # Adicionar nós e arestas ao grafo
    for i in range(n):
        for j in range(i + 1, n):
            if (
                similarity_matrix[i, j] > threshold
            ):  # Apenas conexões com similaridade > 0.7
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Plotar o grafo
    # pos = nx.spring_layout(G)  # Layout para organizar os nós
    pos = nx.kamada_kawai_layout(G)
    weights = nx.get_edge_attributes(G, "weight").values()
    # fig, ax = plt.subplots(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color=list(weights),
        width=list(weights),
        edge_cmap=plt.cm.Blues,
        node_size=500,
        font_size=8,
        font_weight="bold",
    )
    plt.title("Gráfico de Rede da Matriz de Similaridade", fontsize=16)
    return plt


def plot_bar_similaridade(similarity_matrix):
    mean_similarity = np.mean(similarity_matrix, axis=1)

    # Plotar o gráfico de barras
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(mean_similarity)), mean_similarity, color="skyblue")
    plt.xlabel("Índices")
    plt.ylabel("Similaridade Média")
    plt.title("Similaridade Média por Individuo", fontsize=10)
    plt.xticks(range(len(mean_similarity)))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    return plt


def plot_dendograma(similarity_matrix):
    # Converter a matriz de similaridade em uma matriz de distância
    distance_matrix = 1 - similarity_matrix

    # Calcular o linkage (agrupamento hierárquico)
    Z = linkage(distance_matrix, method="ward")

    # Plotar o dendrograma
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=np.arange(similarity_matrix.shape[0]))
    plt.title("Dendrograma da Matriz de Similaridade", fontsize=16)
    plt.xlabel("Índices")
    plt.ylabel("Distância")
    return plt


def plot_contorno(similarity_matrix):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(similarity_matrix, cmap="viridis", levels=20)
    plt.colorbar(contour, label="Similaridade")
    plt.title("Gráfico de Contorno da Matriz de Similaridade", fontsize=16)
    plt.xlabel("Índices")
    plt.ylabel("Índices")
    return plt


def plot_dispersao(similarity_matrix):
    # Reduzir a dimensionalidade usando MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    X_transformed = mds.fit_transform(
        similarity_matrix
    )  # Converter similaridade em distância

    # Plotar o gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c="blue", s=100, alpha=0.7)
    plt.title("Gráfico de Dispersão com MDS", fontsize=16)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    return plt


def hash_cor(r):
    """Gera uma cor RGB a partir de texto (ex: route_short_name)"""
    texto = str(r["route_short_name"]) + str(r["direction_id"])
    h = hashlib.md5(texto.encode()).hexdigest()
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def desenha_mapa_pydeck(linhas):
    layers = []

    df_tudo = pd.concat(linhas, ignore_index=True)
    df_tudo["cor"] = df_tudo.apply(hash_cor, axis=1)
    df_tudo["tooltip"] = df_tudo.apply(
        lambda x: f"""
            <b>Linha:</b> {x['route_short_name']}<br>
            <b>Sentido:</b> {x['direction_id']}<br>
            <b>Ponto:</b> {x['stop_name']} ({x['stop_id']})<br>
            <b>Sequência:</b> {x['stop_sequence']}
        """,
        axis=1,
    )
    df_tudo["text"] = df_tudo["stop_sequence"].astype(str)

    # --------- PATHLAYER (conecta paradas por linha) ----------
    # paths = []
    segmentos = []
    for rota, grupo in df_tudo.groupby("route_short_name"):
        grupo_ordenado = grupo.sort_values(["direction_id", "stop_sequence"])
        grupo_ordenado["lat_next"] = grupo_ordenado["stop_lat"].shift(-1)
        grupo_ordenado["lon_next"] = grupo_ordenado["stop_lon"].shift(-1)
        grupo_ordenado["tooltip"] = grupo_ordenado.apply(
            lambda x: f"""
                <b>Linha:</b> {x['route_short_name']} - <b>Sentido:</b> {x['direction_id']}
            """,
            axis=1,
        )
        for _, row in grupo_ordenado.iloc[:-1].iterrows():
            path = [
                [row["stop_lon"], row["stop_lat"]],
                [row["lon_next"], row["lat_next"]],
            ]
            segmentos.append(
                {"path": path, "color": row["cor"], "tooltip": row["tooltip"]}
            )

    layers.append(
        pdk.Layer(
            "PathLayer",
            data=segmentos,
            get_path="path",
            get_color="color",
            width_scale=3,
            width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        )
    )

    # --------- SCATTERPLOT LAYER (pontos de parada) ----------
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_tudo,
            get_position="[stop_lon, stop_lat]",
            get_color="cor",
            get_radius=7,
            pickable=True,
            auto_highlight=True,
        )
    )

    # --------- TEXT LAYER (número da sequência) ----------
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=df_tudo,
            get_position="[stop_lon, stop_lat]",
            get_text="text",
            get_color=[0, 0, 0],
            get_size=9,
            get_alignment_baseline="'bottom'",
            pickable=False,
        )
    )

    # --------- VIEW E MAPA CLARO ----------
    center_lat = df_tudo["stop_lat"].mean()
    center_lon = df_tudo["stop_lon"].mean()
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": "{tooltip}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "fontSize": "12px",
            },
        },
    )

    return deck


def plot_routes_map_v2(routes, type='ngrams', vis_seq=False, vis_pontos=False):
    def extract_ngrams(seq, min_len=2):
        ngrams = []
        for n in range(len(seq), min_len - 1, -1):
            for i in range(len(seq) - n + 1):
                ngrams.append(tuple(seq[i:i + n]))
        return ngrams

    def lcs(seq1, seq2):
        dp = [["" for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)]
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + [seq1[i - 1]] if isinstance(dp[i - 1][j - 1], list) else [seq1[i - 1]]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
        return dp[-1][-1] if isinstance(dp[-1][-1], list) else []

    # Concatenar todos os dados
    df_tudo = pd.concat(routes, ignore_index=True)

    # Verifica se a coluna 'clusters' existe para ordenação
    if 'clusters' in df_tudo.columns:
        route_order = df_tudo[['route_short_name', 'clusters']].drop_duplicates().sort_values('clusters')
        route_names = route_order['route_short_name'].tolist()
        route_to_cluster = dict(zip(route_order['route_short_name'], route_order['clusters']))
    else:
        route_names = df_tudo['route_short_name'].unique().tolist()
        route_to_cluster = {r: 0 for r in route_names}  # fallback: todos no cluster 0

    N = len(route_names)
    sequences = []
    for rota in route_names:
        grupo = df_tudo[df_tudo['route_short_name'] == rota]
        seq = grupo.sort_values(["direction_id", "stop_sequence"])['stop_id'].tolist()
        sequences.append(seq)

    # Construção do ordenamento dos nós
    if type == 'ngrams':
        ngram_counter = Counter()
        seq_to_ngrams = defaultdict(list)

        for seq in sequences:
            ngrams = extract_ngrams(seq)
            for ng in ngrams:
                ngram_counter[ng] += 1
                seq_to_ngrams[tuple(seq)].append(ng)

        common_ngrams = {ng for ng, count in ngram_counter.items() if count > 1}
        sorted_ngrams = sorted(list(common_ngrams), key=lambda ng: (-len(ng), -ngram_counter[ng]))

        node_to_col = OrderedDict()
        col_idx = 0
        for ng in sorted_ngrams:
            if all(n not in node_to_col for n in ng):
                for n in ng:
                    if n not in node_to_col:
                        node_to_col[n] = col_idx
                        col_idx += 1
        for seq in sequences:
            for n in seq:
                if n not in node_to_col:
                    node_to_col[n] = col_idx
                    col_idx += 1
    elif type == 'lcs':
        lcs_seq = sequences[0]
        for seq in sequences[1:]:
            lcs_seq = lcs(lcs_seq, seq)
            if not lcs_seq:
                break
        ordered_nodes = list(OrderedDict.fromkeys(lcs_seq))
        for seq in sequences:
            for node in seq:
                if node not in ordered_nodes:
                    ordered_nodes.append(node)
        node_to_col = {node: idx for idx, node in enumerate(ordered_nodes)}
    else:
        raise ValueError("Tipo de análise não suportado. Use 'ngrams' ou 'lcs'.")

    # Cores por cluster
    unique_clusters = sorted(set(route_to_cluster.values()))
    color_map = {cl: plt.cm.tab10(i % 10) for i, cl in enumerate(unique_clusters)}

    # Plotagem
    fig, ax = plt.subplots(figsize=(min(0.25 * len(node_to_col), 30), 0.5 * N))
    imp = set()

    for row_idx, (rota, seq) in enumerate(zip(route_names, sequences)):
        cluster = route_to_cluster.get(rota, 0)
        color = color_map[cluster]

        for k, node in enumerate(seq):
            col_idx = node_to_col[node]
            if (col_idx, row_idx) not in imp:
                ax.add_patch(Rectangle((col_idx, -row_idx), 1, 1, color=color))
                if vis_seq:
                    ax.text(col_idx + 0.3, -row_idx + 0.3, str(k+1), ha='center', va='center', fontsize=8, color='white')
                imp.add((col_idx, row_idx))

    # Eixos
    ax.set_xlim(0, len(node_to_col))
    ax.set_ylim(-N, 1)
    if vis_pontos:
        ax.set_xticks(range(len(node_to_col)))
        ax.set_xticklabels(list(node_to_col.keys()), fontsize=8, rotation=90)
    else:
        ax.set_xticks([])
    ax.set_yticks([-i for i in range(N)])
    ax.set_yticklabels(route_names, fontsize=8)
    ax.set_title("Sequências Comuns por Cluster", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_routes_map(routes, type='ngrams', vis_seq=False, vis_pontos=False):
    # Função para extrair todos os n-gramas (subsequências contínuas)
    def extract_ngrams(seq, min_len=2):
        ngrams = []
        for n in range(len(seq), min_len - 1, -1):
            for i in range(len(seq) - n + 1):
                ngrams.append(tuple(seq[i:i + n]))
        return ngrams

    def lcs(seq1, seq2):
        dp = [["" for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)]
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + [seq1[i - 1]] if isinstance(dp[i - 1][j - 1], list) else [seq1[i - 1]]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
        return dp[-1][-1] if isinstance(dp[-1][-1], list) else []

    # Parâmetros
    df_tudo = pd.concat(routes, ignore_index=True)
    if type == 'ngrams':
        # Contar n-gramas em todas as sequências
        ngram_counter = Counter()
        seq_to_ngrams = defaultdict(list)
        route_names = []
        sequences = []
        N = 0
        for rota, grupo in df_tudo.groupby("route_short_name"):
            route_names.append(rota)
            N += 1
            seq = grupo.sort_values(["direction_id", "stop_sequence"])['stop_id'].tolist()
            sequences.append(seq)
            ngrams = extract_ngrams(seq)
            for ng in ngrams:
                ngram_counter[ng] += 1
                seq_to_ngrams[tuple(seq)].append(ng)
        # Selecionar apenas n-gramas que aparecem em mais de uma rota
        common_ngrams = {ng for ng, count in ngram_counter.items() if count > 1}

        # Ordenar os n-gramas por tamanho e frequência (maiores e mais frequentes primeiro)
        sorted_ngrams = sorted(
            list(common_ngrams),
            key=lambda ng: (-len(ng), -ngram_counter[ng])
        )

        # Construir ordenamento baseado nos n-gramas
        # col_used = set()
        node_to_col = OrderedDict()
        col_idx = 0

        for ng in sorted_ngrams:
            can_place = all(n not in node_to_col for n in ng)
            if can_place:
                for n in ng:
                    if n not in node_to_col:
                        node_to_col[n] = col_idx
                        col_idx += 1

        # Colocar os demais nós (não participantes de subsequências comuns)
        for seq in sequences:
            for n in seq:
                if n not in node_to_col:
                    node_to_col[n] = col_idx
                    col_idx += 1
    elif type == 'lcs':
        route_names = []
        sequences = []
        N = 0
        for rota, grupo in df_tudo.groupby("route_short_name"):
            route_names.append(rota)
            N += 1
            seq = grupo.sort_values(["direction_id", "stop_sequence"])['stop_id'].tolist()
            sequences.append(seq)

        # Encontrar LCS geral entre todas as rotas (acumulativa)
        lcs_seq = sequences[0]
        for seq in sequences[1:]:
            lcs_seq = lcs(lcs_seq, seq)
            if not lcs_seq:
                break

        # Construir ordenamento: LCS primeiro, depois o restante
        ordered_nodes = list(OrderedDict.fromkeys(lcs_seq))  # mantém ordem
        for seq in sequences:
            for node in seq:
                if node not in ordered_nodes:
                    ordered_nodes.append(node)

        node_to_col = {node: idx for idx, node in enumerate(ordered_nodes)}
    else:
        raise ValueError("Tipo de análise não suportado. Use 'ngrams' ou 'lcs'.")

    # Plotagem
    fig, ax = plt.subplots(figsize=(min(0.25 * len(node_to_col), 30), 0.5 * N))
    imp = []
    cores = ['steelblue', 'orange', 'green', 'red', 'purple', 'pink', 'brown', 'gray']
    for row_idx, seq in enumerate(sequences):
        k = 0
        for node in seq:
            k += 1
            if k == 1:
                cor = cores[3]
            elif k == len(seq) - 1:
                cor = cores[3]
            else:
                cor = cores[0]
            col_idx = node_to_col[node]
            if (col_idx, row_idx) not in imp:                            
                ax.add_patch(Rectangle((col_idx, -row_idx), 1, 1, color=cor))
                if vis_seq:                    
                    ax.text(col_idx + 0.3, -row_idx + 0.3, str(k), ha='center', va='center', fontsize=8, color='white')
                imp.append((col_idx, row_idx))

    # Eixos
    ax.set_xlim(0, len(node_to_col))
    ax.set_ylim(-N, 1)
    if vis_pontos:
        ax.set_xticks([i for i in range(len(node_to_col))])
        ax.set_xticklabels(node_to_col, fontsize=8, rotation=90)
    else:
        ax.set_xticks([])
    ax.set_yticks([-i for i in range(N)])
    ax.set_yticklabels(route_names, fontsize=8)
    ax.set_title("Sequências Comuns", fontsize=14)
    plt.tight_layout()
    plt.show()


def get_kmeans_inertias(lista_grafos, max_clusters=10, similarity_metrics=None):
    if similarity_metrics is None:
        raise ValueError("similarity_metrics deve ser uma lista de funções de similaridade.")

    inertia_results = {}
    n_graphs = len(lista_grafos)

    for metric in similarity_metrics:
        inertia = []
        X_sim = measures.matriz_similaridade(lista_grafos, func_similaridade=similarity_metrics[metric])
        for k in range(1, min(max_clusters, n_graphs) + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X_sim)
            inertia.append(kmeans.inertia_)

        inertia_results[metric if isinstance(metric, str) else metric.__name__] = inertia
    return inertia_results


def get_skfuzzy_silueta(lista_grafos, max_clusters=10, min_clusters=2, similarity_metrics=None):
    if max_clusters >= len(lista_grafos):
        max_clusters = len(lista_grafos) - 1
    # Inicializar a semente
    np.random.seed(42)
    for metric in similarity_metrics:
        skfuzzy_scores = {}
        siluetas_score = []
        fpc_scores = []
        melhor_score = -np.inf
        melhor_k = None
        X_sim = measures.matriz_similaridade(lista_grafos, func_similaridade=similarity_metrics[metric])
        list_clusters = []
        for k in range(min_clusters, max_clusters + 1):
            c, m, _, d, jm, p, fpc = skfuzzy.cmeans(X_sim.T, k, 1.5, error=0.005, maxiter=1000, init=None)
            try:
                fpc_scores.append(fpc)
                silueta = silhouette_score(X_sim, m.argmax(axis=0))
                siluetas_score.append(silueta)
                list_clusters.append(m)
                # Salvar o melhor score e a matriz de pertinência correspondente
                if fpc > melhor_score:
                    melhor_score = fpc
                    melhor_k = m

            except Exception as e:
                print(f"Erro ao calcular o Silhouette para {k} clusters: {e}")
        skfuzzy_scores[metric] = [siluetas_score, fpc_scores, list_clusters, melhor_score, melhor_k]
    return skfuzzy_scores


# Função para determinar o número ideal de clusters usando o método do cotovelo
def elbow_method_kmeans(lista_grafos, ncols=3, max_clusters=10, similarity_metrics=None):    
    inertias = get_kmeans_inertias(lista_grafos, max_clusters, similarity_metrics)
    n_graphs = len(lista_grafos)
    num_metrics = len(similarity_metrics)

    # Definindo um layout de 2 linhas e 3 colunas
    nrows = (num_metrics // ncols) + (num_metrics % ncols > 0)  # Calcula o número de linhas necessário

    # Criando os subgráficos
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))

    # Desempacotando o array de subgráficos para um formato mais fácil de usar
    axs = axs.flatten()

    cluster_counts = range(1, min(max_clusters, n_graphs) + 1)

    # Plotando os gráficos
    for i, metric in enumerate(similarity_metrics):
        axs[i].plot(cluster_counts, inertias[metric], marker='o')
        axs[i].set_title(f'Métrica: {metric}')
        axs[i].set_xlabel('Número de Clusters')

    # Ajustando o layout
    plt.tight_layout()
    plt.show()


def elbow_method_fuzzy(lista_grafos, ncols=3, max_clusters=10, similarity_metrics=None):    
    skfuzzy_scores = get_skfuzzy_silueta(lista_grafos, max_clusters, similarity_metrics)
    n_graphs = len(lista_grafos)
    num_metrics = len(similarity_metrics)

    # Definindo um layout de 2 linhas e 3 colunas
    nrows = (num_metrics // ncols) + (num_metrics % ncols > 0)  # Calcula o número de linhas necessário

    # Criando os subgráficos
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))

    # Desempacotando o array de subgráficos para um formato mais fácil de usar
    axs = axs.flatten()

    cluster_counts = range(1, min(max_clusters, n_graphs) + 1)

    # Plotando os gráficos
    for i, metric in enumerate(similarity_metrics):
        axs[i].plot(cluster_counts, skfuzzy_scores[metric][0], marker='o')
        axs[i].plot(cluster_counts, skfuzzy_scores[metric][1], marker='o')
        axs[i].set_title(f'Métrica: {metric}')
        axs[i].set_xlabel('Número de Clusters')

    # Ajustando o layout
    plt.tight_layout()
    plt.show()