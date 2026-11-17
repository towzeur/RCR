"""Reliability-Calibrated Routing.

Feature-frame transport, replay-based class radii, and combined ETF/radial
scores. Scoring saved features does not require the training framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RCRConfig:
    """Fixed RCR hyperparameters shared by the reported datasets."""

    radius_power: float = 0.25
    alpha_power: float = 0.5
    alpha_min: float = 0.0
    alpha_max: float = 0.95
    radius_min: float = 1e-4
    radius_max: float = 10.0
    ridge: float = 1e-3


def normalize_if_needed(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    return F.normalize(x.float(), dim=1) if normalize else x.float()


def class_centroids(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_start: int,
    class_end: int,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-class centroids for labels in ``[class_start, class_end)``."""

    features = normalize_if_needed(features, normalize=normalize)
    labels = labels.long()
    dim = features.size(1)
    n_classes = int(class_end - class_start)
    sums = torch.zeros(n_classes, dim, device=features.device, dtype=features.dtype)
    counts = torch.zeros(n_classes, device=features.device, dtype=features.dtype)
    mask = (labels >= class_start) & (labels < class_end)
    if torch.any(mask):
        local = labels[mask] - int(class_start)
        sums.index_add_(0, local, features[mask])
        counts.index_add_(0, local, torch.ones_like(local, dtype=features.dtype))
    centroids = sums / counts.clamp_min(1).unsqueeze(1)
    centroids = normalize_if_needed(centroids, normalize=normalize)
    return centroids, counts


def fit_linear_transport(
    old_features: torch.Tensor,
    current_features: torch.Tensor,
    ridge: float = 1e-3,
    affine: bool = False,
) -> torch.Tensor:
    """Fit a ridge old-to-current feature-frame map."""

    x = old_features.float()
    y = current_features.float()
    if affine:
        x = torch.cat([x, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)], dim=1)
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype)
    if affine:
        eye[-1, -1] = 0.0
    lhs = x.T @ x + float(ridge) * eye
    rhs = x.T @ y
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        return torch.linalg.pinv(lhs) @ rhs


def apply_transport(prototypes: torch.Tensor, transport: torch.Tensor, affine: bool = False) -> torch.Tensor:
    """Apply a fitted transport map to class prototypes."""

    proto = prototypes.float()
    if affine:
        proto = torch.cat([proto, torch.ones(proto.size(0), 1, device=proto.device, dtype=proto.dtype)], dim=1)
    return proto @ transport


def transported_prototypes(
    previous_prototypes: torch.Tensor,
    old_pair_features: torch.Tensor | None,
    current_pair_features: torch.Tensor,
    current_features: torch.Tensor,
    current_labels: torch.Tensor,
    current_class_start: int,
    current_class_end: int,
    config: RCRConfig = RCRConfig(),
    affine: bool = False,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RCR prototypes in the current feature frame.

    Old prototypes are transported with a fitted old-to-current map.  New-class
    prototypes are current-backbone centroids.
    """

    previous = normalize_if_needed(previous_prototypes, normalize=normalize)
    if previous.numel() > 0 and old_pair_features is not None:
        old_pairs = normalize_if_needed(old_pair_features, normalize=normalize)
        current_pairs = normalize_if_needed(current_pair_features, normalize=normalize)
        transport = fit_linear_transport(old_pairs, current_pairs, ridge=config.ridge, affine=affine)
        old_proto = apply_transport(previous, transport, affine=affine)
        old_proto = normalize_if_needed(old_proto, normalize=normalize)
    else:
        old_proto = previous

    new_proto, new_counts = class_centroids(
        current_features,
        current_labels,
        class_start=current_class_start,
        class_end=current_class_end,
        normalize=normalize,
    )
    return torch.cat([old_proto, new_proto], dim=0), new_counts


def class_radius_from_replay(
    replay_features: torch.Tensor,
    replay_labels: torch.Tensor,
    prototypes: torch.Tensor,
    class_num: int,
    config: RCRConfig = RCRConfig(),
    normalize: bool = True,
) -> torch.Tensor:
    """Estimate raw per-class radii from replay cosine residuals."""

    z = normalize_if_needed(replay_features, normalize=normalize)
    proto = normalize_if_needed(prototypes.float()[:class_num], normalize=normalize)
    labels = replay_labels.long()
    valid = (labels >= 0) & (labels < class_num)
    if not torch.any(valid):
        return torch.ones(class_num, device=proto.device, dtype=proto.dtype)
    dist = (2.0 - 2.0 * (z[valid] * proto[labels[valid]]).sum(dim=1)).clamp_min(0.0)
    global_radius = dist.mean().clamp(config.radius_min, config.radius_max)
    radius = torch.full((class_num,), float(global_radius.item()), device=proto.device, dtype=proto.dtype)
    for cls in labels[valid].unique():
        mask = labels[valid] == cls
        radius[int(cls.item())] = dist[mask].mean().clamp(config.radius_min, config.radius_max)
    return radius


def sample_standardize(scores: torch.Tensor) -> torch.Tensor:
    """Standardize logits per sample before mixing heterogeneous scores."""

    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (scores - mean) / std


def fixed_lambda(classes_per_task: int) -> float:
    """Dataset-independent ETF/radial mixture weight."""

    return classes_per_task / (classes_per_task + 1)


def competitive_radius(radius: torch.Tensor, config: RCRConfig = RCRConfig()) -> torch.Tensor:
    """Convert raw class radii into budget-preserving competitive radii."""

    radius = radius.float().clamp(config.radius_min, config.radius_max)
    mean = radius.mean().clamp_min(1e-8)
    deviation = (radius / mean).clamp_min(1e-8).pow(config.radius_power)
    radius = mean * deviation / deviation.mean().clamp_min(1e-8)
    return radius.clamp(config.radius_min, config.radius_max)


def nc_shrink_prototypes(
    prototypes: torch.Tensor,
    etf_vertices: torch.Tensor,
    radius: torch.Tensor,
    config: RCRConfig = RCRConfig(),
) -> torch.Tensor:
    """Retract transported prototypes toward ETF vertices under small radii."""

    mean_radius = radius.mean().clamp_min(1e-8)
    alpha = (mean_radius / radius.clamp_min(1e-8)).pow(config.alpha_power)
    alpha = alpha.clamp(config.alpha_min, config.alpha_max)
    mixed = alpha[:, None] * prototypes + (1.0 - alpha)[:, None] * etf_vertices
    return F.normalize(mixed, dim=1)


def cosine_distance_scores(features: torch.Tensor, prototypes: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Radial reliability logits based on cosine distance."""

    z = F.normalize(features.float(), dim=1)
    proto = F.normalize(prototypes.float(), dim=1)
    dist = (2.0 - 2.0 * (z @ proto.T)).clamp_min(0.0)
    return -dist / radius.view(1, -1)


def rcr_scores(
    features: torch.Tensor,
    etf_vertices: torch.Tensor,
    transported_prototypes: torch.Tensor,
    raw_radius: torch.Tensor,
    classes_per_task: int,
    etf_scores: torch.Tensor | None = None,
    config: RCRConfig = RCRConfig(),
) -> torch.Tensor:
    """Compute the RCR score matrix.

    Args:
        features: Feature matrix with shape ``[N, D]``.
        etf_vertices: Current ETF vertices with shape ``[C, D]``.
        transported_prototypes: Drift-compensated class prototypes ``[C, D]``.
        raw_radius: Replay-estimated class radii ``[C]``.
        classes_per_task: Number of classes introduced at each task.
        etf_scores: Optional precomputed ETF logits ``[N, C]``.
        config: RCR constants.
    """

    class_num = int(etf_vertices.size(0))
    vertices = F.normalize(etf_vertices.float()[:class_num], dim=1)
    proto = F.normalize(transported_prototypes.float()[:class_num], dim=1)
    radius = competitive_radius(raw_radius.float()[:class_num], config=config)
    proto = nc_shrink_prototypes(proto, vertices, radius, config=config)

    if etf_scores is None:
        base = features.float() @ vertices.T
    else:
        base = etf_scores.float()[:, :class_num]
    radial = cosine_distance_scores(features, proto, radius)
    lam = fixed_lambda(int(classes_per_task))
    return (1.0 - lam) * sample_standardize(base) + lam * sample_standardize(radial)


def etf_vertices_from_artifact(artifact: dict, split: dict) -> torch.Tensor:
    """Load ETF vertices, or recover them from features and saved ETF scores."""

    class_num = int(artifact["class_num"])
    etf = artifact.get("etf")
    if etf is not None:
        return F.normalize(etf.float()[:, :class_num].T, dim=1)

    features = split.get("features")
    scores = split.get("etf_scores")
    if features is None or scores is None:
        raise ValueError("RCR needs artifact ETF vertices or saved ETF scores plus features.")
    solution = torch.linalg.lstsq(features.float(), scores.float()).solution
    return F.normalize(solution.T, dim=1)


def rcr_scores_from_artifact(
    artifact: dict,
    split: dict,
    config: RCRConfig = RCRConfig(),
) -> torch.Tensor:
    """Compute RCR from a saved ``artifact_task*.pt`` payload."""

    class_num = int(artifact["class_num"])
    features = split["features"].float()
    vertices = etf_vertices_from_artifact(artifact, split)
    prototypes = artifact["ldc_proto"].float()[:class_num]
    radius = artifact["ldcrad_class_radius"].float()[:class_num]
    return rcr_scores(
        features=features,
        etf_vertices=vertices,
        transported_prototypes=prototypes,
        raw_radius=radius,
        classes_per_task=int(artifact["classes_per_task"]),
        etf_scores=split.get("etf_scores"),
        config=config,
    )


def load_inference_checkpoint(path, device="cpu"):
    """Load a released final checkpoint without unpickling training objects.

    Returns a ResNet-18 in eval mode and a dictionary containing its ETF,
    transported prototypes, radii, and training configuration. Input images
    must use the dataset's evaluation preprocessing; see the dataset modules.
    """
    import sys
    from pathlib import Path

    framework = str(Path(__file__).resolve().parent)
    if framework not in sys.path:
        sys.path.insert(0, framework)
    from backbone.ResNetBlock import resnet18

    state = torch.load(path, weights_only=True, map_location=device)
    if state.get("format_version") != 1:
        raise ValueError("Expected an RCR inference checkpoint (format_version=1)")
    backbone = resnet18(num_classes=int(state["num_classes"]),
                        num_filters=int(state["num_filters"]))
    backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    return backbone.to(device).eval(), state
