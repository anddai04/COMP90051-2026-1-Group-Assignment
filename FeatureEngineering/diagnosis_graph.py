"""
A class file for the diagnosis graph
TODO: maybe split this out... big class
"""

import math
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx

UNKNOWN = "Unknown"

class DiagnosisGraph:
    """
    Diagnosis co occurrence graph class

    Assumptions:
    - diag_1, diag_2, diag_3 have already been preprocessed to ICD-9 stems i.e., 250.7 --> 250
    - Example inputs:
        "250", "428", "414", "038", "V45", "E849"
    - Missing / invalid diagnosis values are already mapped to "Unknown".
    - "Unknown" diagnoses are ignored during graph construction.
    - V/E codes are mapped to "external_causes", not "other".

    Using this in CV:
        graph = DiagnosisGraph(...)
        X_train_graph, X_valid_graph = graph.fit_transform_fold(
            X_train_fold,
            X_valid_fold
        )
        or use fit_transform_augmented which will combine with already constructed features

    Do not fit this graph once on the full dataset before cross-validation!!!
    """

    DIAGNOSIS_GROUPS = [
        # Original major Strack groups
        "circulatory",
        "respiratory",
        "digestive",
        "diabetes",
        "injury",
        "musculoskeletal",
        "genitourinary",
        "neoplasms",

        # Subgroups of "other"
        "symptoms_signs",
        "endocrine_metabolic_other",
        "skin_subcutaneous",
        "infectious_parasitic",
        "mental_disorders",
        "blood_diseases",
        "nervous_system",
        "pregnancy_childbirth",
        "sense_organs",
        "congenital_anomalies",
        "external_causes",

        # Everything else
        "other",
    ]

    def __init__(
        self,
        diag_cols=("diag_1", "diag_2", "diag_3"),
        use_ppmi=True,
        unknown_token=UNKNOWN,
        min_pair_count=1,
        weighted_edges=False,
        role_edge_weights=None,
        weighted_nodes=False,
        role_node_weights=None,
        eps=1e-12,
    ):
        self.diag_cols = tuple(diag_cols)
        self.use_ppmi = bool(use_ppmi)
        self.min_pair_count = int(min_pair_count)
        self.weighted_edges = bool(weighted_edges)
        self.weighted_nodes = bool(weighted_nodes)
        self.eps = eps # small numerical constant to prevent division by zero and log(0)
        self.unknown_token = unknown_token

        self.role_edge_weights = role_edge_weights or {
            (0, 1): 1.00,   # diag_1 -- diag_2
            (0, 2): 0.75,   # diag_1 -- diag_3
            (1, 2): 0.50,   # diag_2 -- diag_3
        }

        self.role_node_weights = role_node_weights or {
            0: 0.50,        # diag_1
            1: 0.30,        # diag_2
            2: 0.20,        # diag_3
        }

        self.is_fitted_ = False

    # helpers
    def is_unknown(self, value):
        if pd.isna(value):
            return True

        s = str(value).strip().lower()

        unknown_values = {
            "",
            "?",
            "nan",
            "none",
            "missing",
            str(self.unknown_token).strip().lower(),
        }

        return s in unknown_values

    def numeric_stem(self, prefix):
        """
        Converts preprocessed numeric ICD-9 stem to int.

        Examples:
            "250" -> 250
            "038" -> 38

        V/E codes and unknown values return None.
        """
        if self.is_unknown(prefix):
            return None

        s = str(prefix).strip().upper()

        if s.startswith("V") or s.startswith("E"):
            return None

        digits = "".join(ch for ch in s if ch.isdigit())

        if digits == "":
            return None

        try:
            return int(digits[:3])
        except ValueError:
            return None

    def diagnosis_group_from_prefix(self, prefix):
        """
        Maps a ICD-9 stem to an a diagnostic group description from Strack
        Unknown values are returned as unknown_token and then ignored
        V/E codes are valid diagnosis-related codes but are not clinical
        disease chapters, so they are mapped to "external_causes".
        """
        if self.is_unknown(prefix):
            return self.unknown_token

        s = str(prefix).strip().upper()

        if s.startswith("V") or s.startswith("E"):
            return "external_causes"

        n = self.numeric_stem(s)

        if n is None:
            return "other"

        # Major groups
        if n == 250:
            return "diabetes"

        if 390 <= n <= 459 or n == 785:
            return "circulatory"

        if 460 <= n <= 519 or n == 786:
            return "respiratory"

        if 520 <= n <= 579 or n == 787:
            return "digestive"

        if 800 <= n <= 999:
            return "injury"

        if 710 <= n <= 739:
            return "musculoskeletal"

        if 580 <= n <= 629 or n == 788:
            return "genitourinary"

        if 140 <= n <= 239:
            return "neoplasms"

        # "Oter" subgroups
        if n in {780, 781, 784} or 790 <= n <= 799:
            return "symptoms_signs"

        if 240 <= n <= 279:
            return "endocrine_metabolic_other"

        if 680 <= n <= 709 or n == 782:
            return "skin_subcutaneous"

        if 1 <= n <= 139:
            return "infectious_parasitic"

        if 290 <= n <= 319:
            return "mental_disorders"

        if 280 <= n <= 289:
            return "blood_diseases"

        if 320 <= n <= 359:
            return "nervous_system"

        if 630 <= n <= 679:
            return "pregnancy_childbirth"

        if 360 <= n <= 389:
            return "sense_organs"

        if 740 <= n <= 759:
            return "congenital_anomalies"

        # Anything not explicitly mapped is "other".
        return "other"

    def row_to_groups(self, row_values):
        return [self.diagnosis_group_from_prefix(x) for x in row_values]

    def valid_group(self, group):
        return not self.is_unknown(group)

    @staticmethod
    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def get_edge_weight(self, i, j):
        if self.weighted_edges:
            return float(self.role_edge_weights.get((i, j), 1.0))
        return 1.0

    def get_node_weight(self, pos):
        if self.weighted_nodes:
            return float(self.role_node_weights.get(pos, 1.0))
        return 1.0

    @staticmethod
    def summary_stats(prefix, values):
        values = np.asarray(list(values), dtype=float)

        if values.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_sum": 0.0,
            }

        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_max": float(values.max()),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_sum": float(values.sum()),
        }

    # Fit graph
    def fit(self, X, y=None):
        """
        Fits the diagnosis-group graph on X
        In cross-validation, X must be the training fold only
        """
        missing_cols = [col for col in self.diag_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing diagnosis columns: {missing_cols}")

        self.node_counts_ = Counter() # for each group
        self.edge_counts_ = Counter() # for each group pair
        self.endpoint_counts_ = Counter() # how often a group appears as an endpoint

        self.total_node_weight_ = 0.0
        self.total_edge_weight_ = 0.0

        diag_values = X.loc[:, list(self.diag_cols)].itertuples(index=False, name=None)

        for row_values in diag_values:
            groups = self.row_to_groups(row_values)

            # Count group occurrences by diagnosis position
            for pos, group in enumerate(groups):
                if self.valid_group(group):
                    w = self.get_node_weight(pos)
                    self.node_counts_[group] += w
                    self.total_node_weight_ += w

            # count pairwise co-occurrences between groups
            for i, j in itertools.combinations(range(len(groups)), 2):
                a, b = groups[i], groups[j]

                if not self.valid_group(a) or not self.valid_group(b):
                    continue

                # Do not create self-loops.
                # Same-group pairs are captured later by same_group features
                if a == b:
                    continue

                w = self.get_edge_weight(i, j)
                key = self.edge_key(a, b)

                self.edge_counts_[key] += w
                self.endpoint_counts_[a] += w
                self.endpoint_counts_[b] += w
                self.total_edge_weight_ += w

        # Fixed group ordering gives stable feature output across folds.
        self.nodes_ = list(self.DIAGNOSIS_GROUPS)

        self.edge_strengths_ = self._compute_edge_strengths()
        self.graph_ = self._build_graph()
        self._compute_centralities()

        self.is_fitted_ = True
        return self

    def _compute_edge_strengths(self):
        edge_strengths = {}

        if self.total_edge_weight_ <= 0:
            return edge_strengths

        for (a, b), c_ab in self.edge_counts_.items():
            if c_ab < self.min_pair_count:
                continue

            if self.use_ppmi:
                p_ab = c_ab / max(self.total_edge_weight_, self.eps)
                p_a = self.endpoint_counts_[a] / max(2.0 * self.total_edge_weight_, self.eps)
                p_b = self.endpoint_counts_[b] / max(2.0 * self.total_edge_weight_, self.eps)

                pmi = math.log((p_ab + self.eps) / (p_a * p_b + self.eps))
                strength = max(pmi, 0.0)
            else:
                strength = float(c_ab)

            if strength > 0:
                edge_strengths[(a, b)] = strength

        return edge_strengths

    def _build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes_)

        for (a, b), strength in self.edge_strengths_.items():
            G.add_edge(a, b, weight=strength)

        return G

    def _compute_centralities(self):
        self.weighted_degree_ = dict(self.graph_.degree(weight="weight"))

        if self.graph_.number_of_edges() > 0:
            self.pagerank_ = nx.pagerank(self.graph_, weight="weight")
            self.clustering_ = nx.clustering(self.graph_, weight="weight")
        else:
            self.pagerank_ = {node: 0.0 for node in self.nodes_}
            self.clustering_ = {node: 0.0 for node in self.nodes_}

    # Transform data into graph features
    def transform(self, X):
        """
        Transforms X into graph features using the fitted graph.
        """
        if not self.is_fitted_:
            raise RuntimeError("DiagnosisGraph must be fitted before calling transform().")

        missing_cols = [col for col in self.diag_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing diagnosis columns: {missing_cols}")

        records = []
        diag_values = X.loc[:, list(self.diag_cols)].itertuples(index=False, name=None)

        for row_values in diag_values:
            groups = self.row_to_groups(row_values)
            valid_groups = [g for g in groups if self.valid_group(g)]

            rec = {}

            # Basic encounter-level group features
            rec["diag_graph_num_valid_diag"] = len(valid_groups)
            rec["diag_graph_num_unknown_diag"] = len(groups) - len(valid_groups)
            rec["diag_graph_num_unique_groups"] = len(set(valid_groups))

            # Diagnosis-role features
            for pos, group in enumerate(groups):
                role = f"diag{pos + 1}"

                is_unknown = not self.valid_group(group)
                rec[f"{role}_unknown"] = int(is_unknown)

                # Diagnosis group indicators for the row
                # e.g., diag1_group_is_diabetes, diag2_group_is_circulatory
                for g in self.DIAGNOSIS_GROUPS:
                    rec[f"{role}_group_is_{g}"] = int((not is_unknown) and group == g)

                # Fitted graph node lookups
                if is_unknown:
                    count = 0.0
                    degree = 0.0
                    pagerank = 0.0
                    clustering = 0.0
                else:
                    count = self.node_counts_.get(group, 0.0)
                    degree = self.weighted_degree_.get(group, 0.0)
                    pagerank = self.pagerank_.get(group, 0.0)
                    clustering = self.clustering_.get(group, 0.0)

                rec[f"{role}_group_log_count"] = math.log1p(count)
                rec[f"{role}_group_weighted_degree"] = degree
                rec[f"{role}_group_pagerank"] = pagerank
                rec[f"{role}_group_clustering"] = clustering

            # Summary of node-level graph properties.
            node_log_counts = [
                math.log1p(self.node_counts_.get(g, 0.0))
                for g in valid_groups
            ]

            node_degrees = [
                self.weighted_degree_.get(g, 0.0)
                for g in valid_groups
            ]

            node_pageranks = [
                self.pagerank_.get(g, 0.0)
                for g in valid_groups
            ]

            node_clusterings = [
                self.clustering_.get(g, 0.0)
                for g in valid_groups
            ]

            rec.update(self.summary_stats("diag_graph_group_log_count", node_log_counts))
            rec.update(self.summary_stats("diag_graph_group_degree", node_degrees))
            rec.update(self.summary_stats("diag_graph_group_pagerank", node_pageranks))
            rec.update(self.summary_stats("diag_graph_group_clustering", node_clusterings))

            # Pairwise edge features
            pair_names = {
                (0, 1): "12",
                (0, 2): "13",
                (1, 2): "23",
            }

            pair_strengths = []
            pair_log_counts = []
            pair_seen = []
            pair_unknown_or_same = []

            for i, j in itertools.combinations(range(len(groups)), 2):
                pair_name = pair_names[(i, j)]
                a, b = groups[i], groups[j]

                if not self.valid_group(a) or not self.valid_group(b):
                    raw_count = 0.0
                    strength = 0.0
                    seen = 0
                    unknown_or_same = 1
                    same_group = 0
                elif a == b:
                    raw_count = 0.0
                    strength = 0.0
                    seen = 0
                    unknown_or_same = 1
                    same_group = 1
                else:
                    key = self.edge_key(a, b)
                    raw_count = self.edge_counts_.get(key, 0.0)
                    strength = self.edge_strengths_.get(key, 0.0)
                    seen = int(key in self.edge_strengths_)
                    unknown_or_same = 0
                    same_group = 0

                rec[f"diag_graph_pair_{pair_name}_strength"] = strength
                rec[f"diag_graph_pair_{pair_name}_log_count"] = math.log1p(raw_count)
                rec[f"diag_graph_pair_{pair_name}_seen"] = seen
                rec[f"diag_graph_pair_{pair_name}_unknown_or_same"] = unknown_or_same
                rec[f"diag_graph_pair_{pair_name}_same_group"] = same_group

                pair_strengths.append(strength)
                pair_log_counts.append(math.log1p(raw_count))
                pair_seen.append(seen)
                pair_unknown_or_same.append(unknown_or_same)

            rec.update(self.summary_stats("diag_graph_pair_strength", pair_strengths))
            rec.update(self.summary_stats("diag_graph_pair_log_count", pair_log_counts))

            rec["diag_graph_num_seen_pairs"] = int(np.sum(pair_seen))
            rec["diag_graph_num_unknown_or_same_pairs"] = int(np.sum(pair_unknown_or_same))
            rec["diag_graph_num_unseen_valid_pairs"] = int(
                np.sum([
                    1
                    for strength, unknown_or_same in zip(pair_strengths, pair_unknown_or_same)
                    if unknown_or_same == 0 and strength == 0.0
                ])
            )

            records.append(rec)

        return pd.DataFrame(records, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def fit_transform_fold(self, X_train_fold, X_valid_fold, y_train_fold=None):
        """
        Fits on X_train_fold only, then transforms train and validation/test
        """
        self.fit(X_train_fold, y=y_train_fold)
        return self.transform(X_train_fold), self.transform(X_valid_fold)

    def fit_transform_fold_augmented(
        self,
        X_train_fold,
        X_valid_fold,
        y_train_fold=None,
        drop_diag_cols=True,
        reset_index=True,
    ):
        """
        Fold-safe graph construction plus concatenation with existing features

        If doing OHE:
            1. Fit graph on train fold only
            2. Fit one-hot encoder on train fold only
            3. Concatenate graph features and one-hot features
        """
        X_train_graph, X_valid_graph = self.fit_transform_fold(
            X_train_fold=X_train_fold,
            X_valid_fold=X_valid_fold,
            y_train_fold=y_train_fold,
        )

        if drop_diag_cols:
            X_train_base = X_train_fold.drop(columns=list(self.diag_cols))
            X_valid_base = X_valid_fold.drop(columns=list(self.diag_cols))
        else:
            X_train_base = X_train_fold.copy()
            X_valid_base = X_valid_fold.copy()

        if reset_index:
            X_train_aug = pd.concat(
                [
                    X_train_base.reset_index(drop=True),
                    X_train_graph.reset_index(drop=True),
                ],
                axis=1,
            )
            X_valid_aug = pd.concat(
                [
                    X_valid_base.reset_index(drop=True),
                    X_valid_graph.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            X_train_aug = pd.concat([X_train_base, X_train_graph], axis=1)
            X_valid_aug = pd.concat([X_valid_base, X_valid_graph], axis=1)

        return X_train_aug, X_valid_aug

    # Diagnostics
    def graph_summary(self):
        if not self.is_fitted_:
            raise RuntimeError("DiagnosisGraph must be fitted before calling graph_summary().")

        return {
            "graph_type": "expanded_diagnosis_group_graph",
            "diag_cols": self.diag_cols,
            "unknown_token": self.unknown_token,
            "use_ppmi": self.use_ppmi,
            "min_pair_count": self.min_pair_count,
            "weighted_edges": self.weighted_edges,
            "weighted_nodes": self.weighted_nodes,
            "num_nodes": self.graph_.number_of_nodes(),
            "num_raw_edges": len(self.edge_counts_),
            "num_retained_edges": self.graph_.number_of_edges(),
            "total_node_weight": self.total_node_weight_,
            "total_edge_weight": self.total_edge_weight_,
        }

    def get_node_table(self):
        if not self.is_fitted_:
            raise RuntimeError("DiagnosisGraph must be fitted before calling get_node_table().")

        rows = []

        for node in self.nodes_:
            rows.append({
                "group": node,
                "node_count": self.node_counts_.get(node, 0.0),
                "endpoint_count": self.endpoint_counts_.get(node, 0.0),
                "weighted_degree": self.weighted_degree_.get(node, 0.0),
                "pagerank": self.pagerank_.get(node, 0.0),
                "clustering": self.clustering_.get(node, 0.0),
            })

        return pd.DataFrame(rows).sort_values("node_count", ascending=False)

    def get_edge_table(self):
        if not self.is_fitted_:
            raise RuntimeError("DiagnosisGraph must be fitted before calling get_edge_table().")

        rows = []

        for (a, b), raw_count in self.edge_counts_.items():
            rows.append({
                "group_a": a,
                "group_b": b,
                "raw_count": raw_count,
                "retained": int((a, b) in self.edge_strengths_),
                "edge_strength": self.edge_strengths_.get((a, b), 0.0),
            })

        if len(rows) == 0:
            return pd.DataFrame(
                columns=["group_a", "group_b", "raw_count", "retained", "edge_strength"]
            )

        return pd.DataFrame(rows).sort_values("raw_count", ascending=False)