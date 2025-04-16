# Carregando Bibliotegas e Funções
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
import networkx as nx
import pickle
import duckdb
import zipfile
import matplotlib.pyplot as plt
from geopy.distance import great_circle


def detectar_sentido(pontos_info, ponto_a, ponto_b):

    pontos = [p for p in pontos_info if p is not None]

    if len(pontos) < 2:
        return "0"  # Não há dados suficientes para detectar sentido

    # Calcular o vetor médio de movimento entre os pontos
    vetor_total = [0, 0]
    for i in range(1, len(pontos)):
        delta_lat = pontos[i]["stop_lat"] - pontos[i - 1]["stop_lat"]
        delta_lon = pontos[i]["stop_lon"] - pontos[i - 1]["stop_lon"]
        vetor_total[0] += delta_lat
        vetor_total[1] += delta_lon

    # Média do vetor de movimento
    vetor_medio = (
        vetor_total[0] / (len(pontos) - 1),
        vetor_total[1] / (len(pontos) - 1),
    )

    # Vetor da linha A -> B
    vetor_ab = (
        ponto_b[0] - ponto_a[0],
        ponto_b[1] - ponto_a[1],
    )

    # Produto escalar entre vetor de movimento e vetor AB
    produto_escalar = vetor_medio[0] * vetor_ab[0] + vetor_medio[1] * vetor_ab[1]

    if produto_escalar >= 0:
        return "1"  # Indo no sentido A -> B
    else:
        return "0"  # Indo no sentido B -> A


def inclui_sentido(df):
    tp = (
        df[df["stop_sequence"] == 1]["stop_lat"].values[0],
        df[df["stop_sequence"] == 1]["stop_lon"].values[0],
    )  # ponto inicial
    max_seq = df["stop_sequence"].max()
    ts = (
        df[df["stop_sequence"] == max_seq]["stop_lat"].values[0],
        df[df["stop_sequence"] == max_seq]["stop_lon"].values[0],
    )  # ponto final
    df["sentido_detectado"] = df.apply(
        lambda row: detectar_sentido(
            [
                row,
                df.iloc[row.name - 1] if row.name > 0 else None,
                df.iloc[row.name - 2] if row.name > 1 else None,
            ],
            tp,
            ts,
        ),
        axis=1,
    )
    return df


# Pega Pontos da Linha do GPS
def carregar_paradas_por_rota(gtfs_path, route_short_name="TODAS"):
    from core.commons import ler_csv_de_zip

    con = duckdb.connect()
    # Lê os arquivos principais
    route_short_name = route_short_name.upper()
    dtype = {
        "route_id": str,
        "route_short_name": str,
        "route_long_name": str,
        "stop_sequence": int,
        "direction_id": int,
        "stop_id": str,
        "stop_name": str,
        "zone_id": str,
        "stop_lat": float,
        "stop_lon": float,
    }
    if gtfs_path.lower().endswith(".zip"):
        if not os.path.isfile(gtfs_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {gtfs_path}")
        # gtfs_path = os.path.abspath(gtfs_path)
        # gtfs_path = os.path.abspath(gtfs_path).replace("\\", "/")
        trips_path = ler_csv_de_zip(gtfs_path, "trips.txt")
        routes_path = ler_csv_de_zip(gtfs_path, "routes.txt")
        stop_times_path = ler_csv_de_zip(gtfs_path, "stop_times.txt")
        stops_path = ler_csv_de_zip(gtfs_path, "stops.txt")
        query = f""" 
            SELECT distinct 
                r."route_id" route_id,
                r."route_short_name", 
                r."route_long_name",
                s."stop_sequence",
                t."direction_id", 
                CAST(s."stop_id" AS VARCHAR) stop_id, 
                st."stop_name",
                st."zone_id",
                st."stop_lat",
                st."stop_lon" 
            FROM trips_path t
                inner join routes_path r on t."route_id" = r."route_id"
                inner join stop_times_path s on t."trip_id" = s."trip_id"
                inner join stops_path st on CAST(s."stop_id" AS VARCHAR) = CAST(st."stop_id" AS VARCHAR)
            WHERE (r."route_short_name" = '{route_short_name}') or ('{route_short_name}' = 'TODAS')
                order by r."route_short_name",s."stop_sequence"
        """
    else:
        trips_path = f"{gtfs_path}/trips.txt"
        routes_path = f"{gtfs_path}/routes.txt"
        stop_times_path = f"{gtfs_path}/stop_times.txt"
        stops_path = f"{gtfs_path}/stops.txt"
        query = f""" 
            SELECT distinct 
                r."route_id",
                r."route_short_name", 
                r."route_long_name",
                s."stop_sequence",
                t."direction_id", 
                CAST(s."stop_id" AS VARCHAR) stop_id, 
                st."stop_name",
                st."zone_id",
                st."stop_lat",
                st."stop_lon" 
            FROM read_csv_auto('{trips_path}') t
                inner join read_csv_auto('{routes_path}') r on t."route_id" = r."route_id"
                inner join read_csv_auto('{stop_times_path}') s on t."trip_id" = s."trip_id"
                inner join read_csv_auto('{stops_path}') st on CAST(s."stop_id" AS VARCHAR) = CAST(st."stop_id" AS VARCHAR)
            WHERE (r."route_short_name" = '{route_short_name}') or ('{route_short_name}' = 'TODAS')
                order by r."route_short_name",s."stop_sequence"
        """

    viagens_rota = con.execute(query).df().astype(dtype)

    if viagens_rota.empty:
        raise ValueError(
            f"Nenhuma viagem encontrada para route_short_name = {route_short_name}"
        )

    resultado = viagens_rota.reset_index(drop=True)

    return resultado


# Pega Pontos da Linha do stun
def stun_paradas_por_rota(trips, route_short_name="TODAS"):
    con = duckdb.connect()
    df_trips = pd.read_parquet(trips, engine="pyarrow")
    con.register("trips", df_trips)
    dtype = {
        "route_id": str,
        "route_short_name": str,
        "route_long_name": str,
        "stop_sequence": int,
        "direction_id": int,
        "stop_id": str,
        "stop_name": str,
        "zone_id": str,
        "stop_lat": float,
        "stop_lon": float,
    }
    route_short_name = route_short_name.upper()
    query = f""" 
        SELECT distinct 
            t."CODIGO_LINHA" route_id,
            t."CODIGO_LINHA" route_short_name, 
            t."CODIGO_LINHA" route_long_name,
            t."ORDEM_PONTO" stop_sequence, 
            CASE 
                WHEN t."SENTIDO_VIAGEM" = 'I' THEN 0
                WHEN t."SENTIDO_VIAGEM" = 'V' THEN 1
                ELSE 2  -- ou outro valor padrão
            END AS direction_id,
            t."CODIGO_PARADA" stop_id, 
            t."NOME_PARADA" stop_name,
            '' zone_id,
            t."LATITUDE" stop_lat,
            t."LONGITUDE" stop_lon
        FROM trips t
        WHERE (t."CODIGO_LINHA" = '{route_short_name}') or ('{route_short_name}' = 'TODAS')
            order by t."CODIGO_LINHA",t."ORDEM_PONTO"
    """
    viagens_rota = con.execute(query).df().astype(dtype)

    if viagens_rota.empty:
        raise ValueError(
            f"Nenhuma viagem encontrada para route_short_name = {route_short_name}"
        )

    resultado = viagens_rota.reset_index(drop=True)

    return resultado


def rotas_to_grafo(df):

    G = nx.DiGraph()
    # Agrupar por trip_id para conectar paradas em sequência
    for route_id, grupo in df.groupby("route_id"):
        grupo = grupo.sort_values(["direction_id", "stop_sequence"])
        for i in range(len(grupo) - 1):
            origem = grupo.iloc[i]
            destino = grupo.iloc[i + 1]

            origem_id = origem["stop_id"]
            destino_id = destino["stop_id"]

            # Adiciona nós (caso ainda não existam)
            if origem_id not in G:
                G.add_node(
                    origem_id,
                    stop_name=origem["stop_name"],
                    direction_id=origem["direction_id"],
                    pos=(origem["stop_lat"], origem["stop_lon"]),
                    lat=origem["stop_lat"],
                    lon=origem["stop_lon"],
                )

            if destino_id not in G:
                G.add_node(
                    destino_id,
                    stop_name=destino["stop_name"],
                    direction_id=destino["direction_id"],
                    pos=(origem["stop_lat"], origem["stop_lon"]),
                    lat=destino["stop_lat"],
                    lon=destino["stop_lon"],
                )

            # Calcular distância geográfica (peso opcional)
            origem_coord = (origem["stop_lat"], origem["stop_lon"])
            destino_coord = (destino["stop_lat"], destino["stop_lon"])
            distancia = great_circle(origem_coord, destino_coord).kilometers

            # Adiciona aresta com atributos
            if (origem_id, destino_id) not in G.edges:
                G.add_edge(
                    origem_id,
                    destino_id,
                    route_short_name=origem["route_short_name"],
                    direction_id=origem["direction_id"],
                    route_id=route_id,
                    distancia=distancia,
                    weight=distancia,
                )

    return G


def show_grafo(G):
    # Para visualizar o grafo, você pode usar o networkx ou outras bibliotecas de visualização

    # Layout para evitar sobreposição
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    cor_por_tipo = {
        0: "green",
        1: "skyblue",
    }
    node_colors = [cor_por_tipo[G.nodes[n]["direction_id"]] for n in G.nodes]
    edge_colors = [cor_por_tipo[G.edges[n]["direction_id"]] for n in G.edges()]
    # Tamanho da figura maior e sem labels para não poluir
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        width=2,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, "stop_name")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black")

    plt.axis("off")
    plt.show()


def carregar_shapes_por_rota(gtfs_path, route_short_name="TODAS"):
    con = duckdb.connect()

    route_short_name = route_short_name.upper()

    if gtfs_path.lower().endswith(".zip"):
        if not os.path.isfile(gtfs_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {gtfs_path}")
        # gtfs_path = os.path.abspath(gtfs_path)
        # gtfs_path = os.path.abspath(gtfs_path).replace("\\", "/")
        trips_path = ler_csv_de_zip(gtfs_path, "trips.txt")
        routes_path = ler_csv_de_zip(gtfs_path, "routes.txt")
        stop_times_path = ler_csv_de_zip(gtfs_path, "stop_times.txt")
        stops_path = ler_csv_de_zip(gtfs_path, "stops.txt")
        shapes_path = ler_csv_de_zip(gtfs_path, "shapes.txt")
        query = f"""
            WITH viagens AS (
                SELECT DISTINCT 
                    r."route_id", 
                    r."route_short_name", 
                    r."route_long_name", 
                    t."trip_id", 
                    t."shape_id",                
                    t."direction_id"
                FROM routes_path r
                    JOIN trips_path t ON t."route_id" = r."route_id"
                WHERE (r."route_short_name" = '{route_short_name}') OR ('{route_short_name}' = 'TODAS')
            ),
            shapes AS (
                SELECT distinct
                    v."route_id",
                    v."route_short_name",
                    v."route_long_name",
                    s."shape_pt_sequence" stop_sequence, 
                    '' stop_name,
                    '' zone_id,
                    v."direction_id",
                    s."shape_id" stop_id,
                    s."shape_pt_lat" stop_lat,
                    s."shape_pt_lon" stop_lon                    
                FROM viagens v
                JOIN shapes_path s ON v."shape_id" = s."shape_id"
            )
            SELECT distinct sp.* FROM shapes sp
            ORDER BY sp."route_short_name", sp."direction_id", sp."stop_sequence", sp."stop_id"
        """
    else:
        query = f"""
            WITH viagens AS (
                SELECT DISTINCT 
                    r."route_id", 
                    r."route_short_name", 
                    r."route_long_name", 
                    t."trip_id", 
                    t."shape_id",                
                    t."direction_id"
                FROM read_csv_auto('{gtfs_path}/routes.txt') r
                JOIN read_csv_auto('{gtfs_path}/trips.txt') t ON t."route_id" = r."route_id"
                WHERE (r."route_short_name" = '{route_short_name}') OR ('{route_short_name}' = 'TODAS')
            ),
            shapes AS (
                SELECT distinct
                    v."route_id",
                    v."route_short_name",
                    v."route_long_name",
                    s."shape_pt_sequence" stop_sequence, 
                    '' stop_name,
                    '' zone_id,
                    v."direction_id",
                    s."shape_id" stop_id,
                    s."shape_pt_lat" stop_lat,
                    s."shape_pt_lon" stop_lon                    
                FROM viagens v
                JOIN read_csv_auto('{gtfs_path}/shapes.txt') s ON v."shape_id" = s."shape_id"
            )
            SELECT distinct * FROM shapes sp
            ORDER BY sp."route_short_name", sp."direction_id", sp."stop_sequence", sp."stop_id"
        """

    shapes_rota = con.execute(query).df()

    if shapes_rota.empty:
        raise ValueError(
            f"Nenhum shape encontrado para route_short_name = {route_short_name}"
        )

    return shapes_rota.reset_index(drop=True)


def carregar_grafos(arquivo_saida):
    # Carregar o dicionário de grafos de um arquivo pickle
    with open(arquivo_saida, "rb") as arquivo:
        grafos = pickle.load(arquivo)
    print(f"Grafos carregados de {arquivo_saida}")
    return grafos


def ler_csv_de_zip(zip_path, nome_arquivo_csv):
    with zipfile.ZipFile(zip_path, "r") as z:
        if nome_arquivo_csv not in z.namelist():
            raise FileNotFoundError(f"{nome_arquivo_csv} não encontrado no .zip")
        with z.open(nome_arquivo_csv) as f:
            df = pd.read_csv(f, low_memory=False)
    return df
