import os

import torch
import torch.nn.functional as F

from models.exorth_ce_svd_ldcncm import extend_orth as LDCNCM
from utils.args import ArgumentParser


class extend_orth(LDCNCM):
    """LDC-NCM with replay-estimated transport uncertainty.

    LDC-NCM treats all transported prototypes as equally reliable.  In practice,
    older task blocks have larger replay residuals in the current feature frame.
    LDC-Rad estimates that residual from the replay buffer and scores classes
    with a heteroscedastic spherical RBF:

        s_c(x) = -d(z, p_c) / r_c - lambda log r_c.

    The radius is estimated only from training-time replay data, so inference is
    single-sample and non-transductive.
    """

    NAME = "exorth_ce_svd_ldcrad"
    COMPATIBILITY = ["class-il", "task-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = LDCNCM.get_parser(parser)
        parser.add_argument("--ldcrad_enable", type=int, default=1)
        parser.add_argument("--ldcrad_mode", type=str, default="task",
                            choices=["task", "class"])
        parser.add_argument("--ldcrad_start_task", type=int, default=8,
                            help="Apply uncertainty scoring when current_task >= this zero-based task index.")
        parser.add_argument("--ldcrad_shrink", type=float, default=0.75,
                            help="Shrink radii toward the global replay residual mean.")
        parser.add_argument("--ldcrad_radius_transform", type=str, default="additive",
                            choices=["additive", "power_budget"],
                            help="additive uses (1-rho)r+rho mean; power_budget preserves mean radius with normalized multiplicative deviations.")
        parser.add_argument("--ldcrad_power", type=float, default=0.25,
                            help="Exponent for --ldcrad_radius_transform power_budget.")
        parser.add_argument("--ldcrad_logterm", type=float, default=0.0,
                            help="Coefficient of -log(radius) normalizer. 0 is pure reliability scaling.")
        parser.add_argument("--ldcrad_score_mix", type=float, default=1.0,
                            help="Final score mix when LDC-Rad is active: 1.0 is pure LDC-Rad, 0.0 is pure auxiliary head.")
        parser.add_argument("--ldcrad_mix_head", type=str, default="etf",
                            choices=["etf", "ce", "ldc"],
                            help="Auxiliary head used when mixing with LDC-Rad. 'etf' preserves the original LDC-Rad behavior, 'ce' uses the trained linear classifier, 'ldc' uses transported NCM scores.")
        parser.add_argument("--ldcrad_mix_norm", type=str, default="none",
                            choices=["none", "sample"],
                            help="Normalize auxiliary-head and LDC-Rad logits before mixing. 'sample' z-scores each sample.")
        parser.add_argument("--ldcrad_mix_rule", type=str, default="fixed",
                            choices=["fixed", "gate_soft", "gate_pick"],
                            help="How to combine auxiliary and LDC-Rad scores. fixed uses ldcrad_score_mix; gates are sample-wise confidence rules.")
        parser.add_argument("--ldcrad_gate_conf", type=str, default="max",
                            choices=["max", "margin", "task_margin"],
                            help="Confidence statistic for gate_soft/gate_pick.")
        parser.add_argument("--ldcrad_gate_temp", type=float, default=2.0,
                            help="Temperature for gate_soft: sigmoid(temp * (conf_rad - conf_aux)).")
        parser.add_argument("--ldcrad_min", type=float, default=1e-4)
        parser.add_argument("--ldcrad_max", type=float, default=10.0)
        parser.add_argument("--ldcrad_report", type=int, default=1)
        parser.add_argument("--ldcrad_artifact_path", type=str, default="",
                            help="Optional .pt path/prefix for paper artifact dumps after each task.")
        parser.add_argument("--ldcrad_artifact_eval", type=int, default=1,
                            help="If 1, include seen test features/scores in artifact dumps.")
        parser.add_argument("--ldcrad_artifact_buffer", type=int, default=1,
                            help="If 1, include replay-buffer features/scores in artifact dumps.")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.register_buffer("ldcrad_class_radius", torch.ones(self.dataset.N_CLASSES))
        self.register_buffer("ldcrad_task_radius", torch.ones(self.dataset.N_TASKS))
        self.register_buffer("ldcrad_ready_classes", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        with torch.no_grad():
            if self._ldcrad_active():
                features = self.net(x, returnt="features")
                z = F.normalize(features, dim=1) if int(getattr(self.args, "ldc_norm_eval", 1)) else features
                proto = self.ldc_proto[:self.class_num].to(z.device, z.dtype)
                if int(getattr(self.args, "ldc_norm_eval", 1)):
                    proto = F.normalize(proto, dim=1)
                sim = z @ proto.T
                dist = (2.0 - 2.0 * sim).clamp_min(0.0)
                radius = self._ldcrad_radius_for_classes(self.class_num, z.device, z.dtype)
                logits = -dist / radius.view(1, -1).clamp_min(float(getattr(self.args, "ldcrad_min", 1e-4)))
                logterm = float(getattr(self.args, "ldcrad_logterm", 0.0))
                if logterm != 0.0:
                    logits = logits - logterm * torch.log(radius).view(1, -1)
                return self._ldcrad_mix_with_aux(features, logits, ldc_scores=sim)
            return super().forward(x)

    def end_task(self, dataset) -> None:
        super().end_task(dataset)
        self._ldcrad_update_radii()
        if getattr(self.args, "ldcrad_artifact_path", ""):
            self._ldcrad_dump_artifacts(dataset)

    def _ldcrad_active(self):
        return (
            int(getattr(self.args, "ldcrad_enable", 1))
            and self._ldc_active()
            and int(self.ldcrad_ready_classes.item()) >= int(self.class_num)
            and self.current_task >= int(getattr(self.args, "ldcrad_start_task", 8))
        )

    @torch.no_grad()
    def _ldcrad_update_radii(self):
        if not self._ldc_active() or self.buffer.is_empty():
            return
        try:
            data = self.buffer.get_all_data(transform=self.normalization_transform, device=self.device)
        except Exception:
            data = self.buffer.get_all_data(device=self.device)
        inputs = data[0]
        labels = data[1].to(self.device, dtype=torch.long)
        valid = (labels >= 0) & (labels < int(self.class_num))
        if not torch.any(valid):
            return
        inputs = inputs[valid]
        labels = labels[valid]

        was_training = self.net.training
        self.net.eval()
        chunks = []
        bs = max(int(getattr(self.args, "batch_size", 32)), 1)
        for chunk in inputs.split(bs):
            feats = self.net(chunk.to(self.device), returnt="features")
            chunks.append(feats.detach())
        if was_training:
            self.net.train()
        if not chunks:
            return
        features = torch.cat(chunks, dim=0)
        z = F.normalize(features, dim=1) if int(getattr(self.args, "ldc_norm_eval", 1)) else features
        proto = self.ldc_proto[:self.class_num].to(self.device)
        if int(getattr(self.args, "ldc_norm_eval", 1)):
            proto = F.normalize(proto, dim=1)
        dist = (2.0 - 2.0 * (z * proto[labels]).sum(dim=1)).clamp_min(0.0)

        min_r = float(getattr(self.args, "ldcrad_min", 1e-4))
        max_r = float(getattr(self.args, "ldcrad_max", 10.0))
        global_r = dist.mean().clamp(min_r, max_r)
        class_r = torch.full((self.class_num,), float(global_r.item()), device=self.device)
        for cls in labels.unique():
            m = labels == cls
            class_r[int(cls.item())] = dist[m].mean().clamp(min_r, max_r)
        task_r = torch.full((self.dataset.N_TASKS,), float(global_r.item()), device=self.device)
        cpt = int(self.dataset.N_CLASSES_PER_TASK)
        seen_tasks = max(1, int(self.class_num) // cpt)
        for task_id in range(seen_tasks):
            m = (labels // cpt) == task_id
            if torch.any(m):
                task_r[task_id] = dist[m].mean().clamp(min_r, max_r)

        self.ldcrad_class_radius[:self.class_num] = class_r.detach().to(self.ldcrad_class_radius.device)
        self.ldcrad_task_radius[:seen_tasks] = task_r[:seen_tasks].detach().to(self.ldcrad_task_radius.device)
        self.ldcrad_ready_classes.fill_(int(self.class_num))

        if int(getattr(self.args, "ldcrad_report", 1)):
            radii = task_r[:seen_tasks]
            print(
                f"LDC-Rad task {self.current_task + 1}: "
                f"mode={getattr(self.args, 'ldcrad_mode', 'task')} "
                f"start={int(getattr(self.args, 'ldcrad_start_task', 8))} "
                f"shrink={float(getattr(self.args, 'ldcrad_shrink', 0.75)):.2f} "
                f"logterm={float(getattr(self.args, 'ldcrad_logterm', 0.0)):.2f} "
                f"task_radius_min={radii.min().item():.4f} "
                f"task_radius_max={radii.max().item():.4f} "
                f"task_radius_mean={radii.mean().item():.4f}",
                flush=True,
            )

    def _ldcrad_radius_for_classes(self, class_num, device, dtype):
        mode = str(getattr(self.args, "ldcrad_mode", "task"))
        if mode == "class":
            radius = self.ldcrad_class_radius[:class_num].to(device=device, dtype=dtype)
        else:
            cpt = int(self.dataset.N_CLASSES_PER_TASK)
            class_tasks = torch.arange(class_num, device=device) // cpt
            radius = self.ldcrad_task_radius.to(device=device, dtype=dtype)[class_tasks]
        shrink = float(getattr(self.args, "ldcrad_shrink", 0.75))
        mean = radius.mean()
        transform = str(getattr(self.args, "ldcrad_radius_transform", "additive"))
        if transform == "power_budget":
            power = float(getattr(self.args, "ldcrad_power", 0.25))
            dev = (radius / mean.clamp_min(1e-8)).clamp_min(1e-8).pow(power)
            radius = mean * dev / dev.mean().clamp_min(1e-8)
        else:
            radius = (1.0 - shrink) * radius + shrink * mean
        return radius.clamp(float(getattr(self.args, "ldcrad_min", 1e-4)),
                            float(getattr(self.args, "ldcrad_max", 10.0)))

    @torch.no_grad()
    def _ldcrad_dump_artifacts(self, dataset):
        if int(self.class_num) <= 0:
            return
        path = self._ldcrad_artifact_task_path(str(self.args.ldcrad_artifact_path))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        cpt = int(self.dataset.N_CLASSES_PER_TASK)
        class_num = int(self.class_num)
        tasks = max(1, class_num // cpt)
        payload = {
            "task": int(self.current_task + 1),
            "class_num": class_num,
            "classes_per_task": cpt,
            "tasks": tasks,
            "etf": self.etf[:, :class_num].detach().cpu() if self.etf is not None else None,
            "ldc_proto": self.ldc_proto[:class_num].detach().cpu(),
            "ldc_counts": self.ldc_counts[:class_num].detach().cpu(),
            "ldcrad_class_radius": self.ldcrad_class_radius[:class_num].detach().cpu(),
            "ldcrad_task_radius": self.ldcrad_task_radius[:tasks].detach().cpu(),
            "ldcrad_ready_classes": int(self.ldcrad_ready_classes.item()),
            "ldcrad_active": bool(self._ldcrad_active()),
            "args": {
                "ldcrad_mode": str(getattr(self.args, "ldcrad_mode", "task")),
                "ldcrad_start_task": int(getattr(self.args, "ldcrad_start_task", 8)),
                "ldcrad_shrink": float(getattr(self.args, "ldcrad_shrink", 0.75)),
                "ldcrad_radius_transform": str(getattr(self.args, "ldcrad_radius_transform", "additive")),
                "ldcrad_power": float(getattr(self.args, "ldcrad_power", 0.25)),
                "ldcrad_logterm": float(getattr(self.args, "ldcrad_logterm", 0.0)),
                "ldcrad_score_mix": float(getattr(self.args, "ldcrad_score_mix", 1.0)),
                "ldcrad_mix_head": str(getattr(self.args, "ldcrad_mix_head", "etf")),
                "ldcrad_mix_norm": str(getattr(self.args, "ldcrad_mix_norm", "none")),
                "ldcrad_mix_rule": str(getattr(self.args, "ldcrad_mix_rule", "fixed")),
                "ldcrad_gate_conf": str(getattr(self.args, "ldcrad_gate_conf", "max")),
                "ldcrad_gate_temp": float(getattr(self.args, "ldcrad_gate_temp", 2.0)),
                "ldc_mode": str(getattr(self.args, "ldc_mode", "")),
                "ldc_fit_source": str(getattr(self.args, "ldc_fit_source", "")),
                "ldc_ridge": float(getattr(self.args, "ldc_ridge", 0.0)),
            },
        }
        if int(getattr(self.args, "ldcrad_artifact_buffer", 1)):
            payload["buffer"] = self._ldcrad_collect_buffer_payload(class_num, tasks, cpt)
        if int(getattr(self.args, "ldcrad_artifact_eval", 1)):
            payload["test"] = self._ldcrad_collect_test_payload(dataset, class_num, tasks, cpt)
        torch.save(payload, path)
        print(f"LDC-Rad artifacts saved to {path}", flush=True)

    def _ldcrad_artifact_task_path(self, path):
        root, ext = os.path.splitext(path)
        if ext != ".pt":
            return f"{path}_task{self.current_task + 1}.pt"
        return f"{root}_task{self.current_task + 1}{ext}"

    @torch.no_grad()
    def _ldcrad_collect_buffer_payload(self, class_num, tasks, cpt):
        if self.buffer.is_empty():
            return {}
        try:
            data = self.buffer.get_all_data(transform=self.normalization_transform, device=self.device)
        except Exception:
            data = self.buffer.get_all_data(device=self.device)
        return self._ldcrad_payload_from_batches(
            [(data[0], data[1].to(self.device, dtype=torch.long))],
            class_num, tasks, cpt, source="buffer",
        )

    @torch.no_grad()
    def _ldcrad_collect_test_payload(self, dataset, class_num, tasks, cpt):
        batches = []
        for loader in getattr(dataset, "test_loaders", [])[:tasks]:
            for data in loader:
                batches.append((data[0], data[1].to(self.device, dtype=torch.long)))
        return self._ldcrad_payload_from_batches(batches, class_num, tasks, cpt, source="test")

    @torch.no_grad()
    def _ldcrad_payload_from_batches(self, batches, class_num, tasks, cpt, source):
        if not batches:
            return {}
        was_training = self.net.training
        self.net.eval()
        bs = max(int(getattr(self.args, "batch_size", 32)), 1)
        feats, labels = [], []
        for inputs, y in batches:
            inputs = inputs.to(self.device)
            y = y.to(self.device, dtype=torch.long)
            for chunk, yy in zip(inputs.split(bs), y.split(bs)):
                feats.append(self.net(chunk, returnt="features").detach().cpu())
                labels.append(yy.detach().cpu())
        if was_training:
            self.net.train()
        if not feats:
            return {}
        features = torch.cat(feats, 0).float()
        labels = torch.cat(labels, 0).long()
        valid = (labels >= 0) & (labels < class_num)
        features = features[valid]
        labels = labels[valid]

        z = F.normalize(features, dim=1) if int(getattr(self.args, "ldc_norm_eval", 1)) else features
        proto = self.ldc_proto[:class_num].detach().cpu().float()
        if int(getattr(self.args, "ldc_norm_eval", 1)):
            proto = F.normalize(proto, dim=1)
        ldc_scores = z @ proto.T
        ce_scores = self._ldcrad_ce_scores(features, class_num)
        etf_scores = self._ldcrad_etf_scores(features, class_num)
        radius = self._ldcrad_radius_for_classes(class_num, z.device, z.dtype).detach().cpu()
        dist = (2.0 - 2.0 * ldc_scores).clamp_min(0.0)
        rad_scores = -dist / radius.view(1, -1).clamp_min(float(getattr(self.args, "ldcrad_min", 1e-4)))
        logterm = float(getattr(self.args, "ldcrad_logterm", 0.0))
        if logterm != 0.0:
            rad_scores = rad_scores - logterm * torch.log(radius).view(1, -1)
        head = str(getattr(self.args, "ldcrad_mix_head", "etf"))
        if head == "ce":
            aux_scores = ce_scores
        elif head == "ldc":
            aux_scores = ldc_scores
        else:
            aux_scores = etf_scores
        mixed_scores = self._ldcrad_mix_scores_offline(rad_scores, aux_scores)
        return {
            "source": source,
            "n": int(features.size(0)),
            "features": features.cpu(),
            "labels": labels.cpu(),
            "true_task": (labels // cpt).cpu(),
            "ldc_scores": ldc_scores.cpu(),
            "ldcrad_scores": rad_scores.cpu(),
            "ce_scores": ce_scores.cpu() if ce_scores is not None else None,
            "etf_scores": etf_scores.cpu() if etf_scores is not None else None,
            "mixed_scores": mixed_scores.cpu() if mixed_scores is not None else None,
            "ldc_pred": ldc_scores.argmax(1).cpu(),
            "ldcrad_pred": rad_scores.argmax(1).cpu(),
            "radius": radius.cpu(),
        }

    def _ldcrad_mix_with_aux(self, features, rad_logits, ldc_scores=None):
        mix = float(getattr(self.args, "ldcrad_score_mix", 1.0))
        rule = str(getattr(self.args, "ldcrad_mix_rule", "fixed"))
        if rule == "fixed" and mix >= 1.0:
            return rad_logits
        aux_logits = self._ldcrad_aux_logits(features, ldc_scores=ldc_scores)
        if aux_logits is None:
            return rad_logits
        if rule == "fixed" and mix <= 0.0:
            return aux_logits
        rad_logits, aux_logits = self._ldcrad_prepare_mix(rad_logits, aux_logits)
        return self._ldcrad_combine_scores(rad_logits, aux_logits)

    def _ldcrad_aux_logits(self, features, ldc_scores=None):
        head = str(getattr(self.args, "ldcrad_mix_head", "etf"))
        if head == "ldc":
            return ldc_scores
        if head == "ce":
            classifier = getattr(self.net, "classifier", None)
            if classifier is None:
                return None
            try:
                return classifier(features)[:, :self.class_num]
            except Exception:
                return None
        if self.etf is None:
            return None
        etf = self.etf[:, :self.class_num].to(features.device, features.dtype)
        return features @ etf

    def _ldcrad_mix_scores_offline(self, rad_scores, aux_scores):
        mix = float(getattr(self.args, "ldcrad_score_mix", 1.0))
        rule = str(getattr(self.args, "ldcrad_mix_rule", "fixed"))
        if aux_scores is None:
            return rad_scores
        if rule == "fixed" and mix >= 1.0:
            return rad_scores
        if rule == "fixed" and mix <= 0.0:
            return aux_scores
        rad_scores, aux_scores = self._ldcrad_prepare_mix(rad_scores, aux_scores)
        return self._ldcrad_combine_scores(rad_scores, aux_scores)

    def _ldcrad_prepare_mix(self, rad_scores, aux_scores):
        mode = str(getattr(self.args, "ldcrad_mix_norm", "none"))
        if mode == "sample":
            rad_scores = self._ldcrad_sample_standardize(rad_scores)
            aux_scores = self._ldcrad_sample_standardize(aux_scores)
        return rad_scores, aux_scores

    def _ldcrad_combine_scores(self, rad_scores, aux_scores):
        rule = str(getattr(self.args, "ldcrad_mix_rule", "fixed"))
        if rule == "gate_pick":
            rad_conf = self._ldcrad_confidence(rad_scores)
            aux_conf = self._ldcrad_confidence(aux_scores)
            return torch.where((rad_conf > aux_conf).view(-1, 1), rad_scores, aux_scores)
        if rule == "gate_soft":
            rad_conf = self._ldcrad_confidence(rad_scores)
            aux_conf = self._ldcrad_confidence(aux_scores)
            temp = float(getattr(self.args, "ldcrad_gate_temp", 2.0))
            weight = torch.sigmoid(temp * (rad_conf - aux_conf)).view(-1, 1)
            return weight * rad_scores + (1.0 - weight) * aux_scores
        mix = float(getattr(self.args, "ldcrad_score_mix", 1.0))
        return mix * rad_scores + (1.0 - mix) * aux_scores

    def _ldcrad_confidence(self, scores):
        kind = str(getattr(self.args, "ldcrad_gate_conf", "max"))
        if kind == "max":
            return scores.max(dim=1).values
        if kind == "task_margin":
            cpt = int(self.dataset.N_CLASSES_PER_TASK)
            tasks = max(1, scores.shape[1] // cpt)
            if tasks < 2:
                return torch.zeros(scores.shape[0], dtype=scores.dtype, device=scores.device)
            task_scores = scores[:, :tasks * cpt].view(scores.shape[0], tasks, cpt).max(dim=2).values
            return self._ldcrad_margin(task_scores)
        return self._ldcrad_margin(scores)

    @staticmethod
    def _ldcrad_margin(scores):
        if scores.shape[1] < 2:
            return torch.zeros(scores.shape[0], dtype=scores.dtype, device=scores.device)
        top2 = scores.topk(2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    @staticmethod
    def _ldcrad_sample_standardize(scores):
        return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    def _ldcrad_ce_scores(self, features, class_num):
        classifier = getattr(self.net, "classifier", None)
        if classifier is None:
            return None
        try:
            scores = classifier(features.to(next(classifier.parameters()).device))
        except Exception:
            return None
        return scores[:, :class_num].detach().cpu()

    def _ldcrad_etf_scores(self, features, class_num):
        etf = getattr(self, "etf", None)
        if etf is None:
            return None
        try:
            scores = features.to(self.device) @ etf[:, :class_num].to(self.device)
        except Exception:
            return None
        return scores.detach().cpu()
