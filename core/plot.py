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
