"""Sparse historical transaction graph plus typed shared-attribute nodes.

Snapshots contain events and observable labels STRICTLY before their cutoff.
Account-to-attribute edges affect communities, but never create projected cliques.
PageRank and fraud exposure use only actual account-to-account transactions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import time

import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse

from fraud_pipeline.kaggle_data import mature_labels

GRAPH_COLUMNS = [
    "sender_degree", "receiver_degree", "number_of_known_fraud_neighbors",
    "fraction_of_neighbors_known_fraud", "fraud_neighbor_exposure",
    "fraud_neighbor_exposure_normalized", "community_risk", "community_risk_normalized",
    "graph_pagerank", "pagerank_normalized", "interaction", "common_neighbors",
    "common_fraud_neighbors", "fraud_neighbor_overlap", "graph_snapshot_age_hours",
]
SUPERVISED_GRAPH_COLUMNS = ["sender_degree", "receiver_degree", "number_of_known_fraud_neighbors", "fraud_neighbor_exposure_normalized"]


@dataclass
class GraphSnapshot:
    cutoff: pd.Timestamp
    accounts: pd.Index
    adjacency: sparse.csr_matrix
    known_fraud: np.ndarray
    features: pd.DataFrame
    edge_attributes: pd.DataFrame
    metadata: dict

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) and (frame.timestamp < self.cutoff).any():
            raise ValueError("Cannot use a future graph snapshot to score an earlier event")
        sender = self.accounts.get_indexer(frame.account_id)
        receiver = self.accounts.get_indexer(frame.counterparty_account_id)
        out = self.features.reindex(frame.account_id).reset_index(drop=True).fillna(0)
        out["receiver_degree"] = self.features.sender_degree.reindex(frame.counterparty_account_id).fillna(0).to_numpy()
        common = np.zeros(len(frame), dtype=np.float32)
        fraud_common = np.zeros(len(frame), dtype=np.float32)
        overlap = np.zeros(len(frame), dtype=np.float32)
        for i, (a, b) in enumerate(zip(sender, receiver)):
            if a < 0 or b < 0:
                continue
            na = self.adjacency.indices[self.adjacency.indptr[a]:self.adjacency.indptr[a + 1]]
            nb = self.adjacency.indices[self.adjacency.indptr[b]:self.adjacency.indptr[b + 1]]
            shared = np.intersect1d(na, nb, assume_unique=True)
            common[i] = len(shared)
            fraud_common[i] = self.known_fraud[shared].sum()
            union_fraud = self.known_fraud[na].sum() + self.known_fraud[nb].sum() - fraud_common[i]
            overlap[i] = fraud_common[i] / union_fraud if union_fraud else 0
        out["common_neighbors"] = common
        out["common_fraud_neighbors"] = fraud_common
        out["fraud_neighbor_overlap"] = overlap
        out["graph_snapshot_age_hours"] = (frame.timestamp - self.cutoff).dt.total_seconds().to_numpy() / 3600
        out["interaction"] = out.community_risk_normalized * out.fraud_neighbor_exposure_normalized
        return out.reindex(columns=GRAPH_COLUMNS, fill_value=0).astype(np.float32)

    def neighbors(self, account: str, limit: int = 10) -> list[dict]:
        idx = self.accounts.get_indexer([account])[0]
        if idx < 0:
            return []
        ids = self.adjacency.indices[self.adjacency.indptr[idx]:self.adjacency.indptr[idx + 1]][:limit]
        return [{"account_id": str(self.accounts[i]), "known_fraud": bool(self.known_fraud[i])} for i in ids]


def build_snapshot(history: pd.DataFrame, cutoff: pd.Timestamp, config: dict) -> GraphSnapshot:
    past = history.loc[history.timestamp < cutoff]
    window = config.get("history_window")
    if window:
        past = past.loc[past.timestamp >= cutoff - pd.Timedelta(window)]
    accounts = pd.Index(pd.unique(pd.concat([past.account_id, past.counterparty_account_id], ignore_index=True)))
    n = len(accounts)
    empty_edges = pd.DataFrame(columns=["source", "target", "transaction_count", "total_amount", "average_amount", "first_transaction_time", "last_transaction_time"])
    if n == 0:
        return GraphSnapshot(cutoff, accounts, sparse.csr_matrix((0, 0)), np.zeros(0), pd.DataFrame(index=accounts, columns=GRAPH_COLUMNS).astype(float), empty_edges, {"accounts": 0, "edges": 0, "cutoff": cutoff.isoformat()})

    a, b = accounts.get_indexer(past.account_id), accounts.get_indexer(past.counterparty_account_id)
    pairs = pd.DataFrame({"source": np.minimum(a, b), "target": np.maximum(a, b), "amount": past.amount.to_numpy(), "timestamp": past.timestamp.to_numpy()})
    pairs = pairs[pairs.source != pairs.target]
    edges = pairs.groupby(["source", "target"], as_index=False, sort=False).agg(
        transaction_count=("amount", "size"), total_amount=("amount", "sum"), average_amount=("amount", "mean"),
        first_transaction_time=("timestamp", "min"), last_transaction_time=("timestamp", "max"),
    )
    endpoints = edges[["source", "target"]].to_numpy(dtype=np.int64)
    counts = edges.transaction_count.to_numpy(dtype=float)
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)), (np.r_[endpoints[:, 0], endpoints[:, 1]], np.r_[endpoints[:, 1], endpoints[:, 0]])), shape=(n, n)).tocsr()
    adjacency.sort_indices()
    degree = np.diff(adjacency.indptr).astype(float)
    graph = ig.Graph(n=n, edges=endpoints, directed=False)
    pagerank = np.asarray(graph.pagerank(weights=counts.tolist() if len(counts) else None), dtype=float)

    # Labels apply to senders; a fraudulent transaction does not prove its receiver is fraudulent.
    mature = history.loc[mature_labels(history, cutoff, config.get("label_maturity", "7d"))]
    fraud_accounts = mature.loc[mature.label == 1, "account_id"].unique()
    known = accounts.isin(fraud_accounts).astype(float)
    exposure = adjacency @ np.divide(known, degree, out=np.zeros(n), where=degree > 0)
    fraud_count = adjacency @ known

    # Add typed attribute nodes. A category such as 'mobile' is never a device identity.
    attribute_edges, attribute_weights = [], []
    node_offset = n
    relation_counts = {}
    chosen = [c for c in config.get("shared_attributes", ["device_id", "location", "merchant_category"]) if c in past]
    if "merchant_id" in past:
        chosen = [c for c in chosen if c != "merchant_category"] + ["merchant_id"]
    for column in chosen:
        memberships = past[["account_id", column]].dropna().drop_duplicates()
        memberships = memberships[~memberships[column].astype(str).isin(["unknown", "nan", ""])]
        frequency = memberships.groupby(column, observed=True).size()
        # Single-account device hashes carry no sharing information and need no attribute node.
        shared_values = frequency[frequency >= 2].index
        memberships = memberships[memberships[column].isin(shared_values)]
        keys = pd.Index(memberships[column].unique())
        src = accounts.get_indexer(memberships.account_id)
        dst = keys.get_indexer(memberships[column]) + node_offset
        attribute_edges.append(np.column_stack([src, dst]))
        # Downweight broad shared locations/categories; no quadratic account clique is formed.
        strength = float(config.get("attribute_weight", 1.0))
        attribute_weights.append(strength / np.sqrt(memberships[column].map(frequency).astype(float).to_numpy()))
        relation_counts[column] = {"nodes": len(keys), "edges": len(src)}
        node_offset += len(keys)
    community_edges = np.vstack([endpoints] + attribute_edges) if attribute_edges else endpoints
    community_weights = np.concatenate([np.log1p(counts)] + attribute_weights)
    community_graph = ig.Graph(n=node_offset, edges=community_edges, directed=False)
    ig.set_random_number_generator(random.Random(int(config.get("seed", 42))))
    algorithm = config.get("community_algorithm", "leiden")
    try:
        if not len(community_edges):
            membership = np.arange(n)
        elif algorithm == "leiden":
            iterations = int(config.get("community_iterations", 2))
            if iterations < 1:
                raise ValueError("community_iterations must be positive (bounded Leiden work)")
            membership = np.asarray(community_graph.community_leiden(objective_function="modularity", weights=community_weights, n_iterations=iterations).membership)[:n]
        elif algorithm == "louvain":
            membership = np.asarray(community_graph.community_multilevel(weights=community_weights).membership)[:n]
        else:
            raise ValueError("community_algorithm must be leiden or louvain")
    finally:
        ig.set_random_number_generator(None)
    sizes = np.bincount(membership)
    frauds = np.bincount(membership, weights=known)
    community_risk = (frauds / np.maximum(sizes, 1))[membership]
    exposure_scale = float(config.get("exposure_scale", 1.0))
    pagerank_scale = float(config.get("pagerank_scale", 1.0))
    if exposure_scale <= 0 or pagerank_scale <= 0:
        raise ValueError("Graph normalization scales must be positive")
    features = pd.DataFrame({
        "sender_degree": degree, "number_of_known_fraud_neighbors": fraud_count,
        "fraction_of_neighbors_known_fraud": np.divide(fraud_count, degree, out=np.zeros(n), where=degree > 0),
        "fraud_neighbor_exposure": exposure,
        "fraud_neighbor_exposure_normalized": exposure / (exposure + exposure_scale),
        "community_risk": community_risk, "community_risk_normalized": community_risk,
        "graph_pagerank": pagerank, "pagerank_normalized": pagerank / (pagerank + pagerank_scale / n),
    }, index=accounts).astype(np.float32)
    return GraphSnapshot(cutoff, accounts, adjacency, known, features, edges, {
        "accounts": n, "edges": len(edges), "attribute_relations": relation_counts,
        "known_fraud_accounts": int(known.sum()), "cutoff": cutoff.isoformat(),
        "community_algorithm": algorithm,
        "pagerank_topology": "transaction_count", "exposure_topology": "direct_transaction_neighbors",
    })


def temporal_graph_features(history: pd.DataFrame, target: pd.DataFrame, config: dict, snapshot_cache: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Replay historical snapshots without mutating state or exposing target labels.

    Only labels explicitly supplied in history may mature. Target labels never enter
    the graph through this function; replay evaluation must pass observed history.
    """
    frequency = config.get("snapshot_frequency", "7d").replace("d", "D")
    cutoffs = target.timestamp.dt.floor(frequency)
    combined = pd.concat([history, target.assign(label=np.nan)], ignore_index=True)
    # History wins for overlapping IDs (for offline replay with explicitly known labels).
    combined = combined.drop_duplicates("transaction_id", keep="first").sort_values("timestamp")
    output = pd.DataFrame(0.0, index=target.index, columns=GRAPH_COLUMNS)
    audit = []
    for cutoff, group in target.groupby(cutoffs, sort=True):
        started = time.perf_counter()
        if config.get("log_progress", False):
            print(f"  Snapshot {cutoff.isoformat()}: {len(group):,} targets", flush=True)
        # Cache only identical historical inputs. Future target events cannot
        # invalidate an earlier snapshot or accidentally reuse a future one.
        past = combined.loc[combined.timestamp < cutoff]
        key = None
        if snapshot_cache is not None:
            digest = hashlib.sha256(pd.util.hash_pandas_object(past, index=False).to_numpy().tobytes()).hexdigest()
            key = (cutoff.isoformat(), digest, repr(sorted(config.items())))
        hit = snapshot_cache is not None and key in snapshot_cache
        snapshot = snapshot_cache[key] if hit else build_snapshot(past, cutoff, config)
        if snapshot_cache is not None and not hit:
            if len(snapshot_cache) >= 2:
                snapshot_cache.pop(next(iter(snapshot_cache)))
            snapshot_cache[key] = snapshot
        output.loc[group.index] = snapshot.transform(group).to_numpy()
        audit.append({**snapshot.metadata, "build_and_transform_seconds": time.perf_counter() - started, "cache_hit": hit})
        if config.get("log_progress", False):
            print(f"  Completed in {time.perf_counter() - started:.1f}s: {snapshot.metadata['accounts']:,} accounts, {snapshot.metadata['edges']:,} transaction edges", flush=True)
    return output.reset_index(drop=True).astype(np.float32), audit
