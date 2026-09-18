from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class GraphFeatureConfig:
    account_col: str = "account_id"
    label_col: str = "label"
    max_attribute_group_size: int = 50
    use_louvain: bool = True


class AccountGraphBuilder:
    def __init__(self, config: GraphFeatureConfig):
        self.config = config
        self.graph: nx.Graph | None = None
        self.account_features_: pd.DataFrame | None = None
        self.global_community_risk_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "AccountGraphBuilder":
        cfg = self.config
        account_col = cfg.account_col
        label_col = cfg.label_col
        graph = nx.Graph()

        accounts = df[account_col].astype(str).unique().tolist()
        graph.add_nodes_from(accounts)

        self._add_transfer_edges(graph, df, account_col)
        for attr in ["device_id", "ip_address", "phone", "email"]:
            if attr in df.columns:
                self._add_shared_attribute_edges(graph, df, account_col, attr)

        account_label = (
            df.groupby(account_col)[label_col]
            .max()
            .astype(float)
            .rename("account_label")
        )

        features = self._compute_graph_features(graph, account_label)
        self.graph = graph
        self.account_features_ = features
        if "community_risk" in features.columns and not features.empty:
            self.global_community_risk_ = float(features["community_risk"].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.account_features_ is None:
            raise RuntimeError("Graph builder must be fit before transform.")

        out = df[[self.config.account_col]].copy()
        feats = self.account_features_.copy()
        out = out.merge(feats, how="left", left_on=self.config.account_col, right_index=True)
        defaults = {
            "graph_degree": 0.0,
            "graph_weighted_degree": 0.0,
            "graph_pagerank": 0.0,
            "graph_betweenness": 0.0,
            "graph_clustering": 0.0,
            "graph_component_size": 1.0,
            "graph_community_id": -1.0,
            "community_risk": self.global_community_risk_,
        }
        for col, value in defaults.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        out = out.drop(columns=[self.config.account_col])
        return out

    def _add_transfer_edges(self, graph: nx.Graph, df: pd.DataFrame, account_col: str) -> None:
        if "counterparty_account_id" not in df.columns:
            return
        rows = df[[account_col, "counterparty_account_id", "amount"]].dropna()
        for row in rows.itertuples(index=False):
            a = str(row[0])
            b = str(row[1])
            if a == b:
                continue
            amount = float(row[2]) if row[2] is not None else 0.0
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += amount
                graph[a][b]["tx_count"] += 1.0
            else:
                graph.add_edge(a, b, weight=amount, tx_count=1.0)

    def _add_shared_attribute_edges(
        self,
        graph: nx.Graph,
        df: pd.DataFrame,
        account_col: str,
        attribute_col: str,
    ) -> None:
        grouped = df[[account_col, attribute_col]].dropna().groupby(attribute_col)[account_col].unique()
        for account_ids in grouped.values:
            ids = [str(x) for x in account_ids if str(x) != "nan"]
            ids = list(dict.fromkeys(ids))
            if len(ids) < 2:
                continue
            if len(ids) > self.config.max_attribute_group_size:
                continue
            for a, b in combinations(ids, 2):
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1.0
                    graph[a][b]["shared_attrs"] = graph[a][b].get("shared_attrs", 0.0) + 1.0
                else:
                    graph.add_edge(a, b, weight=1.0, tx_count=0.0, shared_attrs=1.0)

    def _compute_graph_features(
        self,
        graph: nx.Graph,
        account_label: pd.Series,
    ) -> pd.DataFrame:
        nodes = list(graph.nodes())
        if not nodes:
            return pd.DataFrame()

        degree = dict(graph.degree())
        weighted_degree = dict(graph.degree(weight="weight"))
        pagerank = nx.pagerank(graph, weight="weight")

        if graph.number_of_nodes() <= 5000:
            betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
        else:
            betweenness = dict.fromkeys(nodes, 0.0)

        clustering = nx.clustering(graph, weight="weight")
        component_size: dict[str, float] = {}
        for comp in nx.connected_components(graph):
            size = float(len(comp))
            for node in comp:
                component_size[str(node)] = size

        communities = self._detect_communities(graph)
        node_to_community: dict[str, int] = {}
        for idx, community in enumerate(communities):
            for node in community:
                node_to_community[str(node)] = idx

        community_risk = {}
        for idx, community in enumerate(communities):
            labels = [float(account_label.get(str(node), 0.0)) for node in community]
            community_risk[idx] = float(np.mean(labels)) if labels else 0.0

        data = []
        for node in nodes:
            node_str = str(node)
            community_id = node_to_community.get(node_str, -1)
            data.append(
                {
                    "account_id": node_str,
                    "graph_degree": float(degree.get(node, 0.0)),
                    "graph_weighted_degree": float(weighted_degree.get(node, 0.0)),
                    "graph_pagerank": float(pagerank.get(node, 0.0)),
                    "graph_betweenness": float(betweenness.get(node, 0.0)),
                    "graph_clustering": float(clustering.get(node, 0.0)),
                    "graph_component_size": float(component_size.get(node_str, 1.0)),
                    "graph_community_id": float(community_id),
                    "community_risk": float(community_risk.get(community_id, 0.0)),
                }
            )
        out = pd.DataFrame(data).set_index("account_id")
        return out

    def _detect_communities(self, graph: nx.Graph) -> list[set[str]]:
        if self.config.use_louvain:
            try:
                communities = nx.community.louvain_communities(
                    graph, weight="weight", seed=42
                )
                return [set(map(str, c)) for c in communities]
            except Exception:
                pass

        communities = nx.community.label_propagation_communities(graph)
        return [set(map(str, c)) for c in communities]


def link_prediction_features(
    df: pd.DataFrame,
    graph_builder: AccountGraphBuilder,
    account_col: str = "account_id",
    counterparty_col: str = "counterparty_account_id",
) -> pd.DataFrame:
    if graph_builder.graph is None or counterparty_col not in df.columns:
        return pd.DataFrame(
            {
                "common_neighbors": np.zeros(len(df), dtype=float),
                "jaccard_similarity": np.zeros(len(df), dtype=float),
            }
        )

    g = graph_builder.graph
    common_neighbors = []
    jaccard_scores = []

    for row in df[[account_col, counterparty_col]].itertuples(index=False):
        a = str(row[0])
        b = str(row[1])
        if a not in g or b not in g:
            common_neighbors.append(0.0)
            jaccard_scores.append(0.0)
            continue

        neighbors_a = set(g.neighbors(a))
        neighbors_b = set(g.neighbors(b))
        inter = neighbors_a.intersection(neighbors_b)
        union = neighbors_a.union(neighbors_b)
        common_neighbors.append(float(len(inter)))
        jaccard_scores.append(float(len(inter) / len(union)) if union else 0.0)

    return pd.DataFrame(
        {"common_neighbors": common_neighbors, "jaccard_similarity": jaccard_scores}
    )
