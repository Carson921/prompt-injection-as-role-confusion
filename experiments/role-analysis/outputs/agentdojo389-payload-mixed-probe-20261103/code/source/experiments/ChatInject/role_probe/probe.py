"""Frozen GPT-OSS-120B role probe and read-only pre-o_proj head capture.

The versioned classifiers consume ordered attention-head output vectors, *not*
residual ``output_hidden_states``. Their training representation was float16;
standardization, exact GELU, and probability evaluation use float32.
No model loading, checkpoint modification, or generation is performed here.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


DEFAULT_PROBE_PATH = Path(
    "/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/"
    "local-gptoss120b/systemness-complementary-head-readout/models/new.npz"
)
DEFAULT_PROBE_SHA256 = "ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff"
BASE_MODEL_SHA256 = "27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266"
ROLES = ("system", "user", "cot", "assistant", "tool")
HEAD_IDS = (458, 462, 2045, 460, 459, 2047, 456, 997, 247, 2248)
HEADS_PER_LAYER = HEAD_DIM = 64
FEATURE_DIM = len(HEAD_IDS) * HEAD_DIM
SIX_ROLE_PROBE_PATH = Path(
    "/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/"
    "role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz"
)
SIX_ROLE_PROBE_SHA256 = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"
WARMSTART_PROBE_PATH = Path(
    "/home/wentian/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/"
    "outputs/synthetic-role-warmstart-20261027-6h/checkpoints/best.npz"
)
WARMSTART_PROBE_SHA256 = "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df"
SIX_ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
SIX_ROLE_HEAD_IDS = (1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
                     2179, 2288, 1274, 897, 1305, 2091, 2094, 1930)
SIX_ROLE_FEATURE_DIM = len(SIX_ROLE_HEAD_IDS) * HEAD_DIM
_SIX_ROLE_WEIGHT_SHAPES = {
    "0.weight": (128, SIX_ROLE_FEATURE_DIM), "0.bias": (128,),
    "3.weight": (6, 128), "3.bias": (6,),
}
_SIX_ROLE_MODEL_BINDING = {
    "format": "gpt-oss-120b-native-mxfp4",
    "repository": "openai/gpt-oss-120b",
    "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    "kernel_revision": "9fbc1b48e710ac8abf18ea2650ee690d52dc3859",
    "identity_sha256": "ec2e12096a2f43e8b9323036c5a67d7d864c8541501a906a0966c400b6c31795",
}
_WEIGHT_SHAPES = {
    "base.encoder.0.weight": (128, 512),
    "base.encoder.0.bias": (128,),
    "base.head.weight": (5, 128),
    "base.head.bias": (5,),
    "correction.0.weight": (32, 640),
    "correction.0.bias": (32,),
    "correction.3.weight": (1, 32),
    "correction.3.bias": (1,),
}


def array_sha256(value: np.ndarray) -> str:
    """The training code's exact raw, contiguous array checksum."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _state_sha256(state: dict[str, np.ndarray]) -> str:
    value = [
        dict(name=name, shape=list(array.shape), dtype=str(array.dtype), sha256=array_sha256(array))
        for name, array in sorted(state.items())
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate_features(features: np.ndarray, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    features = np.asarray(features)
    _require(
        features.ndim == 2 and features.shape[1] == feature_dim and features.dtype == np.float16,
        f"Expected the training representation: float16 [tokens, {feature_dim}] pre-o_proj features.",
    )
    _require(bool(np.isfinite(features).all()), "Role probe input contains nonfinite features.")
    return np.ascontiguousarray(features)


class RoleProbe:
    """Portable frozen classifier; defaults pin the user-supplied artifact bytes.

    ``predict_proba`` returns float32 columns in ``roles`` order. The optional
    NumPy evaluator independently implements the same formula for CPU parity
    checks, without using PyTorch or a large language model.
    """

    roles = ROLES
    head_ids = HEAD_IDS
    feature_dim = FEATURE_DIM

    def __init__(
        self,
        path: str | Path = DEFAULT_PROBE_PATH,
        device: str = "cpu",
        expected_sha256: str | None = DEFAULT_PROBE_SHA256,
    ) -> None:
        self.path = Path(path).resolve()
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            _require(self.sha256 == expected_sha256, "Role probe artifact SHA256 differs from the pinned artifact.")
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"]))
            self.mean = archive["mean"].copy()
            self.scale = archive["scale"].copy()
            self.weights = {name[8:]: archive[name].copy() for name in archive.files if name.startswith("weight::")}
        self._validate()
        self.device = str(device)
        self._torch_weights: dict[str, Any] | None = None
        self._torch_mean: Any = None
        self._torch_scale: Any = None

    def _validate(self) -> None:
        expected = dict(
            kind="binary-residual-640", arm="new", dimensions=640, base_dimensions=512,
            base_hidden=128, residual_hidden=32, parameters=20545, delta_formula="4*tanh(g)",
            roles=list(ROLES), selected_head_ids=list(HEAD_IDS), base_model_sha256=BASE_MODEL_SHA256,
        )
        for key, value in expected.items():
            _require(self.metadata.get(key) == value, f"Role probe metadata differs for {key}.")
        _require(set(self.weights) == set(_WEIGHT_SHAPES), "Unexpected role probe weight names.")
        for name, shape in _WEIGHT_SHAPES.items():
            value = self.weights[name]
            _require(value.shape == shape and value.dtype == np.float32 and bool(np.isfinite(value).all()),
                     f"Malformed role probe weight: {name}.")
        for name in ("mean", "scale"):
            value = getattr(self, name)
            _require(value.shape == (640,) and value.dtype == np.float32 and bool(np.isfinite(value).all()),
                     f"Malformed role probe {name}.")
            _require(array_sha256(value) == self.metadata.get(f"{name}_sha256"),
                     f"Role probe {name} checksum differs.")
        _require(bool((self.scale > 0).all()), "Role probe scale must be positive.")
        base = {name[5:]: value for name, value in self.weights.items() if name.startswith("base.")}
        _require(_state_sha256(base) == self.metadata.get("embedded_base_weight_sha256"),
                 "Role probe embedded frozen base checksum differs.")

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(
            path=str(self.path), sha256=self.sha256, roles=list(self.roles),
            selected_head_ids=list(self.head_ids),
            selected_layer_head=[[head // 64, head % 64] for head in self.head_ids],
            feature_shape=[None, 640], capture_point="self_attn.o_proj input",
            captured_dtype="float16", inference_dtype="float32", device=self.device,
            mean_sha256=array_sha256(self.mean), scale_sha256=array_sha256(self.scale),
            weight_sha256={name: array_sha256(value) for name, value in self.weights.items()},
            formula="softmax(base(x[:512]) + system_only(4*tanh(correction(x))))",
            standardization="(float32(float16(head_outputs)) - mean) / scale",
            gelu="exact erf", metadata=self.metadata,
        )

    def _init_torch(self) -> None:
        if self._torch_weights is None:
            import torch
            self._torch_weights = {name: torch.from_numpy(value).to(self.device) for name, value in self.weights.items()}
            self._torch_mean = torch.from_numpy(self.mean).to(self.device)
            self._torch_scale = torch.from_numpy(self.scale).to(self.device)

    def predict_proba(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Evaluate the training formula in PyTorch, with dropout disabled."""
        import torch
        from torch.nn import functional as F
        features = _validate_features(features)
        _require(chunk_size > 0, "chunk_size must be positive.")
        if not len(features):
            return np.empty((0, len(ROLES)), np.float32)
        self._init_torch()
        weights = self._torch_weights
        assert weights is not None
        result = np.empty((len(features), len(ROLES)), np.float32)
        with torch.inference_mode():
            for start in range(0, len(features), chunk_size):
                x = torch.from_numpy(features[start:start + chunk_size]).to(self.device).float()
                x = ((x - self._torch_mean) / self._torch_scale).contiguous()
                _require(bool(torch.isfinite(x).all()), "Nonfinite standardized role probe features.")
                hidden = F.gelu(F.linear(x[:, :512].contiguous(), weights["base.encoder.0.weight"], weights["base.encoder.0.bias"]))
                logits = F.linear(hidden, weights["base.head.weight"], weights["base.head.bias"])
                correction = F.gelu(F.linear(x, weights["correction.0.weight"], weights["correction.0.bias"]))
                delta = 4 * torch.tanh(F.linear(correction, weights["correction.3.weight"], weights["correction.3.bias"]))
                logits[:, 0] += delta[:, 0]
                result[start:start + len(x)] = F.log_softmax(logits, dim=1).exp().cpu().numpy()
        _require(bool(np.isfinite(result).all()) and bool(np.allclose(result.sum(1), 1, atol=2e-6)),
                 "Invalid role probabilities.")
        return result

    def predict_proba_numpy(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Independent exact-erf formula; intended for parity checks on small inputs."""
        features = _validate_features(features)
        _require(chunk_size > 0, "chunk_size must be positive.")
        result = np.empty((len(features), len(ROLES)), np.float32)
        weights = self.weights

        def linear(x: np.ndarray, name: str) -> np.ndarray:
            return x @ weights[name + ".weight"].T + weights[name + ".bias"]

        def gelu(x: np.ndarray) -> np.ndarray:
            erf = np.fromiter((math.erf(float(v) / math.sqrt(2)) for v in x.flat), dtype=np.float32, count=x.size).reshape(x.shape)
            return np.float32(0.5) * x * (np.float32(1) + erf)

        for start in range(0, len(features), chunk_size):
            x = (features[start:start + chunk_size].astype(np.float32) - self.mean) / self.scale
            logits = linear(gelu(linear(x[:, :512], "base.encoder.0")), "base.head")
            delta = np.float32(4) * np.tanh(linear(gelu(linear(x, "correction.0")), "correction.3"))
            logits[:, 0] += delta[:, 0]
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            result[start:start + len(x)] = exponentials / exponentials.sum(axis=1, keepdims=True)
        return result


class SixRoleProbe(RoleProbe):
    """Frozen synthetic-agent v1 six-role MLP, bound to the supplied best.npz.

    The archive embeds its FIT standardizer and ordered sixteen-head selection.
    It is a different architecture from ``RoleProbe``; no residual correction or
    five-class remapping is applied. Passing ``expected_sha256=None`` is useful
    only for validating deliberately malformed fixtures in CPU tests.
    """

    roles = SIX_ROLES
    head_ids = SIX_ROLE_HEAD_IDS
    feature_dim = SIX_ROLE_FEATURE_DIM
    artifact_version = "synthetic-agent-final-v7"
    required_metadata = dict(candidate_id="mlp128-seed17", hidden=128, dropout=0.1,
                             heads=list(SIX_ROLE_HEAD_IDS), feature_dim=SIX_ROLE_FEATURE_DIM,
                             standardizer="FIT_family_weighted_population_std")

    def __init__(
        self,
        path: str | Path = SIX_ROLE_PROBE_PATH,
        device: str = "cpu",
        expected_sha256: str | None = SIX_ROLE_PROBE_SHA256,
    ) -> None:
        self.path = Path(path).resolve()
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            _require(self.sha256 == expected_sha256,
                     "Six-role probe artifact SHA256 differs from the pinned artifact.")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            expected_keys = {"metadata", "mean", "scale", *(
                "weight::" + name for name in _SIX_ROLE_WEIGHT_SHAPES)}
            _require(len(archive.files) == len(expected_keys) and set(archive.files) == expected_keys,
                     "Unexpected or missing six-role probe archive fields.")
            raw_metadata = archive["metadata"]
            _require(raw_metadata.shape == () and raw_metadata.dtype.kind == "U",
                     "Six-role probe metadata must be a scalar JSON string.")
            self.metadata = json.loads(str(raw_metadata))
            self.mean = archive["mean"].copy()
            self.scale = archive["scale"].copy()
            self.weights = {name: archive["weight::" + name].copy() for name in _SIX_ROLE_WEIGHT_SHAPES}
        _require(isinstance(self.metadata, dict), "Six-role probe metadata must be a JSON object.")
        self.feature_dim = self.metadata.get("feature_dim")
        self._validate()
        self.head_ids = tuple(self.metadata["heads"])
        self.feature_dim = self.metadata["feature_dim"]
        self.device = str(device)
        self._torch_weights: dict[str, Any] | None = None
        self._torch_mean: Any = None
        self._torch_scale: Any = None

    def _validate(self) -> None:
        _require(isinstance(self.metadata, dict), "Six-role probe metadata must be a JSON object.")
        expected = dict(
            format_version="synthetic-agent-six-role-probe-v1", roles=list(SIX_ROLES),
            feature_position="pre-o_proj", activation="gelu_exact", engineering_smoke_only=False,
            **self.required_metadata,
        )
        for key, value in expected.items():
            _require(self.metadata.get(key) == value, f"Six-role probe metadata differs for {key}.")
        heads = self.metadata.get("heads")
        _require(isinstance(heads, list) and bool(heads)
                 and all(type(head) is int and 0 <= head < 36 * HEADS_PER_LAYER for head in heads)
                 and len(set(heads)) == len(heads), "Six-role probe heads must be distinct valid ordered IDs.")
        _require(type(self.feature_dim) is int and self.feature_dim == len(heads) * HEAD_DIM,
                 "Six-role probe feature_dim differs from its ordered heads.")
        hidden = self.metadata.get("hidden")
        _require(type(hidden) is int and hidden > 0, "Six-role probe hidden width must be a positive integer.")
        _require(isinstance(self.metadata.get("dropout"), (int, float))
                 and 0 <= self.metadata["dropout"] < 1, "Six-role probe dropout metadata is invalid.")
        for name in ("candidate_id", "standardizer"):
            _require(isinstance(self.metadata.get(name), str) and bool(self.metadata[name]),
                     f"Six-role probe {name} metadata is missing.")
        binding = self.metadata.get("model_binding")
        _require(isinstance(binding, dict), "Missing six-role probe model_binding.")
        for key, value in _SIX_ROLE_MODEL_BINDING.items():
            _require(binding.get(key) == value, f"Six-role probe model_binding differs for {key}.")
        shapes = {"0.weight": (hidden, self.feature_dim), "0.bias": (hidden,),
                  "3.weight": (len(self.roles), hidden), "3.bias": (len(self.roles),)}
        for name, shape in shapes.items():
            value = self.weights[name]
            _require(value.shape == shape and value.dtype == np.float32 and bool(np.isfinite(value).all()),
                     f"Malformed six-role probe weight: {name}.")
        for name in ("mean", "scale"):
            value = getattr(self, name)
            _require(value.shape == (self.feature_dim,) and value.dtype == np.float32
                     and bool(np.isfinite(value).all()), f"Malformed six-role probe {name}.")
        _require(bool((self.scale > 0).all()), "Six-role probe scale must be positive.")

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(
            path=str(self.path), sha256=self.sha256, format_version=self.metadata["format_version"],
            artifact_version=self.artifact_version,
            roles=list(self.roles), selected_head_ids=list(self.head_ids),
            selected_layer_head=[[head // HEADS_PER_LAYER, head % HEADS_PER_LAYER] for head in self.head_ids],
            feature_dim=self.feature_dim, feature_shape=[None, self.feature_dim],
            capture_point="self_attn.o_proj input", captured_dtype="float16", inference_dtype="float32",
            device=self.device, mean_sha256=array_sha256(self.mean), scale_sha256=array_sha256(self.scale),
            weight_sha256={name: array_sha256(value) for name, value in self.weights.items()},
            formula=f"softmax(linear6(gelu_exact(linear{self.metadata['hidden']}(standardized_ordered_head_outputs))))",
            standardization="(float32(float16(head_outputs)) - embedded_FIT_mean) / embedded_FIT_scale",
            gelu="exact erf", dropout_enabled=False, model_binding=self.metadata["model_binding"],
            metadata=self.metadata,
        )

    def predict_proba(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Evaluate the frozen six-class MLP in float32 PyTorch."""
        import torch
        from torch.nn import functional as F

        features = _validate_features(features, self.feature_dim)
        _require(chunk_size > 0, "chunk_size must be positive.")
        result = np.empty((len(features), len(self.roles)), dtype=np.float32)
        if not len(features):
            return result
        self._init_torch()
        weights = self._torch_weights
        assert weights is not None
        with torch.inference_mode():
            for start in range(0, len(features), chunk_size):
                x = torch.from_numpy(features[start:start + chunk_size]).to(self.device).float()
                x = (x - self._torch_mean) / self._torch_scale
                _require(bool(torch.isfinite(x).all()), "Nonfinite standardized role probe features.")
                hidden = F.gelu(F.linear(x, weights["0.weight"], weights["0.bias"]), approximate="none")
                logits = F.linear(hidden, weights["3.weight"], weights["3.bias"])
                result[start:start + len(x)] = F.softmax(logits, dim=1).cpu().numpy()
        _require(bool(np.isfinite(result).all()) and bool(np.allclose(result.sum(1), 1, atol=2e-6)),
                 "Invalid six-role probabilities.")
        return result

    def predict_proba_numpy(self, features: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Independent CPU evaluator with the official export's exact GELU."""
        from scipy.special import erf

        features = _validate_features(features, self.feature_dim)
        _require(chunk_size > 0, "chunk_size must be positive.")
        result = np.empty((len(features), len(self.roles)), dtype=np.float32)
        weights = self.weights
        for start in range(0, len(features), chunk_size):
            x = (features[start:start + chunk_size].astype(np.float32) - self.mean) / self.scale
            _require(bool(np.isfinite(x).all()), "Nonfinite standardized role probe features.")
            hidden = x @ weights["0.weight"].T + weights["0.bias"]
            hidden = hidden * np.float32(0.5) * (1 + erf(hidden / np.float32(math.sqrt(2))))
            logits = hidden @ weights["3.weight"].T + weights["3.bias"]
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            result[start:start + len(x)] = exponentials / exponentials.sum(axis=1, keepdims=True)
        _require(bool(np.isfinite(result).all()) and bool(np.allclose(result.sum(1), 1, atol=2e-6)),
                 "Invalid six-role probabilities.")
        return result


class WarmstartSixRoleProbe(SixRoleProbe):
    """The explicitly selected scratch research checkpoint from October 27."""

    artifact_version = "synthetic-role-warmstart-20261027-6h-scratch"
    required_metadata = {**SixRoleProbe.required_metadata, "candidate_id": "scratch",
                         "standardizer": "unchanged_parent_FIT_mean_scale"}

    def __init__(self, path: str | Path = WARMSTART_PROBE_PATH, device: str = "cpu",
                 expected_sha256: str | None = WARMSTART_PROBE_SHA256) -> None:
        super().__init__(path, device=device, expected_sha256=expected_sha256)

    def _validate(self) -> None:
        super()._validate()
        lineage = self.metadata.get("warm_start", {})
        _require(isinstance(lineage, dict) and lineage.get("parent_probe_sha256") == SIX_ROLE_PROBE_SHA256
                 and lineage.get("initialize_weights") is False and lineage.get("trainable_modules") == "all",
                 "Warmstart scratch probe lineage differs from the frozen export.")


class MetadataSixRoleProbe(SixRoleProbe):
    """CPU cache consumer for future ordered-head exports of the same MLP format.

    Production runtimes use ``load_probe``'s explicit artifact allowlist. This
    reader is for offline rescoring: feature position, roles, backbone binding,
    ordered head IDs, dimensions and finite weights remain strictly validated.
    """

    artifact_version = "metadata-bound-offline-export"
    required_metadata: dict[str, Any] = {}

    def __init__(self, path: str | Path, device: str = "cpu",
                 expected_sha256: str | None = None) -> None:
        super().__init__(path, device=device, expected_sha256=expected_sha256)


def load_probe(path: str | Path, device: str = "cpu") -> RoleProbe:
    """Load only one of the three explicitly supported, byte-pinned artifacts."""
    sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    readers = {DEFAULT_PROBE_SHA256: RoleProbe, SIX_ROLE_PROBE_SHA256: SixRoleProbe,
               WARMSTART_PROBE_SHA256: WarmstartSixRoleProbe}
    _require(sha256 in readers, "Unsupported role probe artifact SHA256.")
    return readers[sha256](path, device=device)


class HeadCollector:
    """Capture ordered per-token heads without changing the model's forward pass.

    Call ``begin(flattened_token_positions)`` before each forward, then ``finish``.
    Flattened positions use row-major ``[batch, sequence]`` order. This supports
    left padding by passing only attention-mask nonzero positions. The collector
    deliberately rejects duplicate layer calls: callers must begin a fresh
    capture for each prefill/decode forward, or disable capture during generation.
    """

    def __init__(self, head_ids: Sequence[int] = HEAD_IDS, feature_dim: int | None = None) -> None:
        raw_heads = np.asarray(head_ids)
        _require(raw_heads.ndim == 1 and bool(raw_heads.size) and raw_heads.dtype.kind in "iu",
                 "Head selection must be a nonempty one-dimensional integer sequence.")
        _require(bool(((raw_heads >= 0) & (raw_heads < 36 * HEADS_PER_LAYER)).all())
                 and len(np.unique(raw_heads)) == len(raw_heads),
                 "Head selection contains duplicates or IDs outside the 36-layer model.")
        self.selected = raw_heads.astype(np.int64, copy=True)
        self.feature_dim = len(self.selected) * HEAD_DIM
        _require(feature_dim is None or feature_dim == self.feature_dim,
                 "Feature dimension differs from the ordered head selection.")
        self.layers = sorted(set((self.selected // HEADS_PER_LAYER).tolist()))
        self.enabled = False
        self.features: np.ndarray | None = None
        self.positions = np.empty(0, np.int64)
        self.seen: set[int] = set()
        self._indices: dict[str, Any] = {}

    def begin(self, positions: Sequence[int] | np.ndarray, enabled: bool = True) -> None:
        raw = np.asarray(positions)
        _require(raw.ndim == 1 and (not raw.size or raw.dtype.kind in "iu"), "Capture positions must be a one-dimensional integer vector.")
        self.positions = raw.astype(np.int64, copy=True)
        _require(len(np.unique(self.positions)) == len(self.positions) and bool((self.positions >= 0).all()),
                 "Capture positions contain duplicates or negative indices.")
        self.enabled = enabled
        self.seen = set()
        self._indices = {}
        self.features = np.full((len(self.positions), self.feature_dim), np.nan, np.float16) if enabled else None

    def hook(self, layer: int) -> Any:
        _require(layer in self.layers, "Hook requested for an unselected layer.")

        def receive(module: Any, args: tuple[Any, ...]) -> None:
            if not self.enabled:
                return None
            import torch
            value = args[0]
            _require(layer not in self.seen and value.ndim == 3 and value.shape[-1] == 4096,
                     "Duplicate layer call or unexpected pre-o_proj activation layout.")
            self.seen.add(layer)
            _require(not len(self.positions) or int(self.positions.max()) < value.shape[0] * value.shape[1],
                     "Capture token position exceeds current forward input.")
            key = str(value.device)
            if key not in self._indices:
                self._indices[key] = torch.as_tensor(self.positions, dtype=torch.long, device=value.device)
            ranks = np.flatnonzero(self.selected // 64 == layer)
            heads = torch.as_tensor(self.selected[ranks] % 64, dtype=torch.long, device=value.device)
            sample = value.detach().reshape(-1, 64, 64).index_select(0, self._indices[key]).index_select(1, heads)
            _require(bool(torch.isfinite(sample).all()) and bool((sample.float().abs() <= np.finfo(np.float16).max).all()),
                     "Nonfinite or out-of-float16-range activation; refusing clipping.")
            columns = (ranks[:, None] * 64 + np.arange(64)).reshape(-1)
            assert self.features is not None
            self.features[:, columns] = sample.to("cpu", torch.float16).numpy().reshape(len(self.positions), len(columns))
            return None

        return receive

    def finish(self) -> np.ndarray:
        _require(self.enabled and self.features is not None, "No active feature capture.")
        _require(self.seen == set(self.layers), "Selected layer hook did not run.")
        assert self.features is not None
        _require(bool(np.isfinite(self.features).all()), "Capture contains unwritten/nonfinite head vectors.")
        result = self.features
        # Generation can now proceed through the installed hooks without capture.
        self.enabled = False
        self._indices = {}
        return result

    def install(self, model: Any) -> list[Any]:
        handles = []
        try:
            for layer in self.layers:
                handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(self.hook(layer)))
            return handles
        except BaseException:
            for handle in handles:
                handle.remove()
            raise

    @contextmanager
    def installed(self, model: Any) -> Iterator[HeadCollector]:
        handles = self.install(model)
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

