import unittest
import networkx as nx
import numpy as np
import core.measures as measures
import core.commons as commons


class TestMedidas(unittest.TestCase):
    def setUp(self):
        self.gtfs_path = 'datasets\GTFS_OTTRANS.zip'
        df_gtfs = commons.carregar_paradas_por_rota(self.gtfs_path, 'Todas')
        linhas_selecionadas = df_gtfs.route_short_name.unique()[0:20]
        # Configurar grafos de exemplo
        self.df_grafos = df_gtfs[df_gtfs['route_short_name'].isin(linhas_selecionadas)].copy()
        self.grafos = commons.rotas_to_grafo(self.df_grafos)
        self.g1 = self.grafos[0]
        self.g2 = self.grafos[1]

    def test_dados(self):
        assert len(self.grafos) == 20

    def test_geodesic_distance(self):
        self.assertIsNotNone(measures.geodesic_distance(self.g1, self.g2))

    def test_weighted_distance(self):
        self.assertIsNotNone(measures.weighted_distance(self.g1, self.g2))

    def test_graph_diameter(self):
        self.assertIsNotNone(measures.diameter_similarity(self.g1, self.g2))

    def test_frechet_distance(self):
        self.assertIsNotNone(measures.frechet_similarity(self.g1, self.g2))

    def test_dtw_distance(self):
        self.assertIsNotNone(measures.dtw_similarity(self.g1, self.g2))

    def test_subgraph_similarity(self):
        self.assertIsNotNone(measures.subgraph_similarity(self.g1, self.g2))

    def test_lcs_distance(self):
        self.assertIsNotNone(measures.lcs_similarity(self.g1, self.g2))

    def test_graph_edit_distance(self):
        self.assertIsNotNone(measures.distancia_edit(self.g1, self.g2))

    def test_levenshtein_distance(self):
        self.assertIsNotNone(measures.levenshtein_similarity(self.g1, self.g2))

    def test_cosine_similarity(self):
        self.assertIsNotNone(measures.cosine_similarity(self.g1, self.g2))

    def test_jaccard_similarity(self):
        self.assertIsNotNone(measures.jaccard_similarity(self.g1, self.g2))

    def test_emd_distance(self):
        self.assertIsNotNone(measures.emd_distance(self.g1, self.g2))

    def test_path_distance(self):
        self.assertIsNotNone(measures.path_distance(self.g1, self.g2))

    def test_graph_kernel(self):
        self.assertIsNotNone(measures.graph_kernel(self.g1, self.g2))

    def test_similarity_haversine(self):
        self.assertIsNotNone(measures.similaridade_grafos(self.g1, self.g2))

    def test_subgraph_matching(self):
        self.assertIsNotNone(measures.subgraph_matching(self.g1, self.g2))

    def test_cne_similarity(self):
        self.assertIsNotNone(measures.cne_similarity(self.g1, self.g2))
        
if __name__ == "__main__":
    unittest.main()