from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SamplerConfig:
    random_seed: int = 42


class SamplerAgent:
    def __init__(self, config: SamplerConfig | None = None) -> None:
        self.config = config or SamplerConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def select_indices(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray | None,
        k: int,
        strategy_hint: str | None = None,
    ) -> list[int]:
        _ = labels
        n = embeddings.shape[0]
        if n == 0:
            return []
        if n <= k:
            return list(range(n))

        strategy = strategy_hint or self._choose_strategy(embeddings)

        if strategy == "random":
            return sorted(self.rng.choice(n, size=k, replace=False).tolist())
        if strategy == "outlier":
            return self._outlier_sample(embeddings, k)
        if strategy == "kmeans":
            result = self._kmeans_sample(embeddings, k)
            if result:
                return result
            return sorted(self.rng.choice(n, size=k, replace=False).tolist())

        return sorted(self.rng.choice(n, size=k, replace=False).tolist())

    def _choose_strategy(self, embeddings: np.ndarray) -> str:
        n = embeddings.shape[0]
        if n <= 200:
            return "random"

        center = embeddings.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(embeddings - center, axis=1)
        threshold = np.quantile(distances, 0.9)
        outlier_ratio = float((distances > threshold).mean())

        if outlier_ratio > 0.1:
            return "outlier"
        return "kmeans"

    def _outlier_sample(self, embeddings: np.ndarray, k: int) -> list[int]:
        center = embeddings.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(embeddings - center, axis=1)

        outlier_count = max(1, k // 2)
        center_count = k - outlier_count

        outlier_idx = np.argsort(-distances)[: outlier_count * 2]
        center_idx = np.argsort(distances)[: max(center_count * 3, 1)]

        picked_outlier = self.rng.choice(outlier_idx, size=outlier_count, replace=False)
        picked_center = self.rng.choice(center_idx, size=center_count, replace=False)
        combined = np.unique(np.concatenate([picked_outlier, picked_center]))

        if combined.shape[0] < k:
            missing = k - combined.shape[0]
            all_idx = np.arange(embeddings.shape[0])
            remain = np.setdiff1d(all_idx, combined)
            extra = self.rng.choice(remain, size=missing, replace=False)
            combined = np.concatenate([combined, extra])

        return sorted(combined.tolist())

    def _kmeans_sample(self, embeddings: np.ndarray, k: int) -> list[int]:
        try:
            from sklearn.cluster import KMeans
        except Exception:
            return []

        n = embeddings.shape[0]
        num_clusters = min(k, max(2, int(np.sqrt(n))))

        model = KMeans(n_clusters=num_clusters, random_state=self.config.random_seed, n_init=10)
        labels = model.fit_predict(embeddings)
        centers = model.cluster_centers_

        selected: list[int] = []
        for cluster_id in range(num_clusters):
            idx = np.where(labels == cluster_id)[0]
            if idx.size == 0:
                continue
            cluster_vectors = embeddings[idx]
            center = centers[cluster_id]
            nearest_local = np.argmin(np.linalg.norm(cluster_vectors - center, axis=1))
            selected.append(int(idx[nearest_local]))

        if len(selected) >= k:
            return sorted(selected[:k])

        missing = k - len(selected)
        pool = np.setdiff1d(np.arange(n), np.array(selected, dtype=int))
        if pool.size > 0:
            extra = self.rng.choice(pool, size=min(missing, pool.size), replace=False)
            selected.extend(extra.tolist())

        return sorted(selected)
