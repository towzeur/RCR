import torch
import torch.nn.functional as F

from models.exorth_ce_svd import extend_orth as ProNC
from utils.args import ArgumentParser


class extend_orth(ProNC):
    """ProNC with linear drift compensated NCM inference.

    The ProNC training objective is unchanged.  After each incremental task, the
    current task provides paired features from the previous and current
    backbones.  A ridge affine map old->new is fit on those pairs and applied to
    stored old class centroids.  Current-task centroids are then estimated from
    the current backbone.  Inference uses nearest class mean in the compensated
    feature space instead of raw ETF vertices.
    """

    NAME = 'exorth_ce_svd_ldcncm'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = ProNC.get_parser(parser)
        parser.add_argument('--ldc_enable', type=int, default=1)
        parser.add_argument('--ldc_eval_ncm', type=int, default=1)
        parser.add_argument('--ldc_mode', type=str, default='affine',
                            choices=['affine', 'linear', 'orthogonal', 'translation',
                                     'local_translation', 'none'])
        parser.add_argument('--ldc_fit_source', type=str, default='current',
                            choices=['current', 'buffer', 'mixed', 'hybrid'],
                            help='Pairs used to fit old->new drift: current task, replay buffer old samples, or both.')
        parser.add_argument('--ldc_hybrid_current_weight', type=float, default=0.5,
                            help='For ldc_fit_source=hybrid, blend transported prototypes from current-only and mixed fits.')
        parser.add_argument('--ldc_fit_weighting', type=str, default='none',
                            choices=['none', 'source', 'class', 'source_class'],
                            help='Optional pair weighting for the drift fit. Defaults to the original unweighted fit.')
        parser.add_argument('--ldc_old_pair_weight', type=float, default=1.0,
                            help='Extra multiplicative weight for replay/old pairs when source weighting is enabled.')
        parser.add_argument('--ldc_ridge', type=float, default=1e-3)
        parser.add_argument('--ldc_robust_iters', type=int, default=0,
                            help='Neutral by default. If >0, refit the linear/affine drift map with Huber-style IRLS.')
        parser.add_argument('--ldc_robust_delta_mult', type=float, default=2.0,
                            help='Robust Huber cutoff as this multiple of the median pair residual.')
        parser.add_argument('--ldc_spectral_rho', type=float, default=1.0,
                            help='For linear/affine LDC, shrink singular values of the fitted linear part toward 1. '
                                 '1 keeps the original map; 0 keeps only the orthogonal polar factor.')
        parser.add_argument('--ldc_spectral_smin', type=float, default=0.0,
                            help='Optional lower clamp for fitted singular values after shrink. 0 disables.')
        parser.add_argument('--ldc_spectral_smax', type=float, default=0.0,
                            help='Optional upper clamp for fitted singular values after shrink. 0 disables.')
        parser.add_argument('--ldc_local_k', type=int, default=32,
                            help='Number of nearest paired features for local_translation drift.')
        parser.add_argument('--ldc_local_temp', type=float, default=0.05,
                            help='Softmax temperature for local_translation drift weights.')
        parser.add_argument('--ldc_transport_shrink', type=float, default=1.0,
                            help='0 keeps old prototypes, 1 applies the full estimated drift.')
        parser.add_argument('--ldc_task_residual', type=float, default=0.0,
                            help='Add a task-common old replay residual after the global drift map. 0 disables it.')
        parser.add_argument('--ldc_proto_anchor', type=float, default=0.0,
                            help='Spherical blend transported LDC prototypes toward ETF vertices. 0 disables it.')
        parser.add_argument('--ldc_proto_anchor_scope', type=str, default='old',
                            choices=['old', 'all'],
                            help='Apply prototype anchoring to old transported classes only or all seen classes.')
        parser.add_argument('--ldc_score_mix', type=float, default=1.0,
                            help='Final score = mix*NCM + (1-mix)*ETF, using normalized features.')
        parser.add_argument('--ldc_buffer_blend', type=float, default=0.0,
                            help='Blend transported old prototypes with current-backbone replay centroids.')
        parser.add_argument('--ldc_norm_pairs', type=int, default=1)
        parser.add_argument('--ldc_norm_eval', type=int, default=1)
        parser.add_argument('--ldc_report', type=int, default=1)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        feature_dim = int(getattr(self.net, 'feature_dim', 512))
        self.register_buffer('ldc_proto', torch.zeros(self.dataset.N_CLASSES, feature_dim))
        self.register_buffer('ldc_counts', torch.zeros(self.dataset.N_CLASSES))
        self.register_buffer('ldc_ready_classes', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        with torch.no_grad():
            if self._ldc_active():
                features = self.net(x, returnt='features')
                if int(getattr(self.args, 'ldc_norm_eval', 1)):
                    features = F.normalize(features, dim=1)
                    proto = F.normalize(self.ldc_proto[:self.class_num].to(features.device, features.dtype), dim=1)
                else:
                    proto = self.ldc_proto[:self.class_num].to(features.device, features.dtype)
                ncm_logits = features @ proto.T
                mix = float(getattr(self.args, 'ldc_score_mix', 1.0))
                if mix >= 1.0 or self.etf is None:
                    return ncm_logits
                etf = self.etf[:, :self.class_num].to(features.device, features.dtype)
                if int(getattr(self.args, 'ldc_norm_eval', 1)):
                    etf = F.normalize(etf, dim=0)
                etf_logits = features @ etf
                if mix <= 0.0:
                    return etf_logits
                return mix * ncm_logits + (1.0 - mix) * etf_logits
            return super().forward(x)

    def end_task(self, dataset) -> None:
        if int(getattr(self.args, 'ldc_enable', 1)):
            self._ldc_update_prototypes(dataset)
        super().end_task(dataset)

    def _ldc_active(self):
        return (
            int(getattr(self.args, 'ldc_eval_ncm', 1))
            and int(self.ldc_ready_classes.item()) > 0
            and self.class_num > 0
        )

    @torch.no_grad()
    def _ldc_update_prototypes(self, dataset):
        loader = getattr(self, 'noval_data_loader', None)
        if loader is None:
            return

        was_training = self.net.training
        self.net.eval()
        if self.old_net is not None:
            self.old_net.eval()

        cpt = int(self.dataset.N_CLASSES_PER_TASK)
        self.class_num = cpt * (self.current_task + 1)
        start = cpt * self.current_task
        end = start + cpt

        old_feats, new_feats, new_labels = self._ldc_collect_pairs(loader, label_min=start, label_max=end)
        if new_feats is None:
            if was_training:
                self.net.train()
            return

        if self.current_task > 0 and self.old_net is not None and old_feats is not None:
            if str(getattr(self.args, 'ldc_fit_source', 'current')) == 'hybrid':
                mixed_old, mixed_new, mixed_labels = self._ldc_collect_pairs(
                    dataset.train_loader, label_min=0, label_max=end)
                if mixed_old is not None and mixed_new is not None:
                    self._ldc_transport_old_prototypes_hybrid(
                        current_old=old_feats, current_new=new_feats,
                        mixed_old=mixed_old, mixed_new=mixed_new,
                        old_limit=start, mixed_labels=mixed_labels)
            else:
                fit_old, fit_new, fit_labels = self._ldc_get_fit_pairs(dataset, current_old=old_feats,
                                                                       current_new=new_feats, start=start, end=end)
                if fit_old is not None and fit_new is not None:
                    self._ldc_transport_old_prototypes(fit_old, fit_new, old_limit=start, labels=fit_labels)

        self._ldc_set_current_prototypes(new_feats, new_labels, start, end)
        self._ldc_blend_old_buffer_centroids(dataset, old_limit=start)
        self._ldc_anchor_prototypes(old_limit=start)
        self.ldc_ready_classes.fill_(self.class_num)

        if int(getattr(self.args, 'ldc_report', 1)):
            ready_old = int((self.ldc_counts[:start] > 0).sum().item())
            ready_new = int((self.ldc_counts[start:end] > 0).sum().item())
            print(
                f"LDC-NCM task {self.current_task + 1}: "
                f"classes={self.class_num} old_ready={ready_old} new_ready={ready_new} "
                f"mode={getattr(self.args, 'ldc_mode', 'affine')} ridge={float(getattr(self.args, 'ldc_ridge', 1e-3)):.1e} "
                f"robust_iters={int(getattr(self.args, 'ldc_robust_iters', 0))} "
                f"robust_delta={float(getattr(self.args, 'ldc_robust_delta_mult', 2.0)):.2f} "
                f"spec_rho={float(getattr(self.args, 'ldc_spectral_rho', 1.0)):.2f} "
                f"spec_min={float(getattr(self.args, 'ldc_spectral_smin', 0.0)):.2f} "
                f"spec_max={float(getattr(self.args, 'ldc_spectral_smax', 0.0)):.2f} "
                f"local_k={int(getattr(self.args, 'ldc_local_k', 32))} "
                f"local_t={float(getattr(self.args, 'ldc_local_temp', 0.05)):.3f} "
                f"fit={getattr(self.args, 'ldc_fit_source', 'current')} "
                f"hybrid_w={float(getattr(self.args, 'ldc_hybrid_current_weight', 0.5)):.2f} "
                f"weight={getattr(self.args, 'ldc_fit_weighting', 'none')} "
                f"old_w={float(getattr(self.args, 'ldc_old_pair_weight', 1.0)):.2f} "
                f"shrink={float(getattr(self.args, 'ldc_transport_shrink', 1.0)):.2f} "
                f"task_res={float(getattr(self.args, 'ldc_task_residual', 0.0)):.2f} "
                f"anchor={float(getattr(self.args, 'ldc_proto_anchor', 0.0)):.2f} "
                f"anchor_scope={getattr(self.args, 'ldc_proto_anchor_scope', 'old')} "
                f"score_mix={float(getattr(self.args, 'ldc_score_mix', 1.0)):.2f} "
                f"buffer_blend={float(getattr(self.args, 'ldc_buffer_blend', 0.0)):.2f}",
                flush=True,
            )

        if was_training:
            self.net.train()

    @torch.no_grad()
    def _ldc_collect_pairs(self, loader, label_min, label_max):
        old_chunks, new_chunks, label_chunks = [], [], []
        for data in loader:
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device, dtype=torch.long)
            mask = (labels >= label_min) & (labels < label_max)
            if not torch.any(mask):
                continue
            inputs = inputs[mask]
            labels = labels[mask]
            new_feat = self.net(inputs, returnt='features')
            if int(getattr(self.args, 'ldc_norm_pairs', 1)):
                new_feat = F.normalize(new_feat, dim=1)
            new_chunks.append(new_feat.detach())
            label_chunks.append(labels.detach())
            if self.old_net is not None:
                old_feat = self.old_net(inputs, returnt='features')
                if int(getattr(self.args, 'ldc_norm_pairs', 1)):
                    old_feat = F.normalize(old_feat, dim=1)
                old_chunks.append(old_feat.detach())

        if not new_chunks:
            return None, None, None
        old_feats = torch.cat(old_chunks, dim=0) if old_chunks else None
        new_feats = torch.cat(new_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        return old_feats, new_feats, labels

    @torch.no_grad()
    def _ldc_get_fit_pairs(self, dataset, current_old, current_new, start, end):
        source = str(getattr(self.args, 'ldc_fit_source', 'current'))
        if source == 'current':
            return current_old, current_new, None

        if source == 'buffer':
            label_min, label_max = 0, start
        else:
            label_min, label_max = 0, end

        mixed_old, mixed_new, mixed_labels = self._ldc_collect_pairs(
            dataset.train_loader, label_min=label_min, label_max=label_max)
        if mixed_old is None or mixed_new is None:
            return current_old, current_new, None
        return mixed_old, mixed_new, mixed_labels

    @torch.no_grad()
    def _ldc_transport_old_prototypes(self, old_feats, new_feats, old_limit, labels=None):
        if old_limit <= 0 or int((self.ldc_counts[:old_limit] > 0).sum().item()) == 0:
            return
        mode = str(getattr(self.args, 'ldc_mode', 'affine'))
        proto = self.ldc_proto[:old_limit].to(self.device)
        valid = self.ldc_counts[:old_limit].to(self.device) > 0

        out, pair_mapped = self._ldc_estimate_transport(
            old_feats=old_feats, new_feats=new_feats, proto=proto, mode=mode,
            labels=labels, old_limit=old_limit)

        out = self._ldc_apply_task_residual(out, pair_mapped, new_feats, labels=labels, old_limit=old_limit)

        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            out = F.normalize(out, dim=1)
        shrink = float(getattr(self.args, 'ldc_transport_shrink', 1.0))
        if shrink < 1.0:
            out = (1.0 - shrink) * proto + shrink * out
            if int(getattr(self.args, 'ldc_norm_eval', 1)):
                out = F.normalize(out, dim=1)
        valid_cpu = valid.detach().cpu()
        proto_block = self.ldc_proto[:old_limit].clone()
        proto_block[valid_cpu] = out.detach().to(proto_block.device)[valid_cpu]
        self.ldc_proto[:old_limit] = proto_block

    @torch.no_grad()
    def _ldc_transport_old_prototypes_hybrid(self, current_old, current_new, mixed_old, mixed_new,
                                             old_limit, mixed_labels=None):
        if old_limit <= 0 or int((self.ldc_counts[:old_limit] > 0).sum().item()) == 0:
            return
        mode = str(getattr(self.args, 'ldc_mode', 'affine'))
        proto = self.ldc_proto[:old_limit].to(self.device)
        valid = self.ldc_counts[:old_limit].to(self.device) > 0

        current_out, _ = self._ldc_estimate_transport(
            old_feats=current_old, new_feats=current_new, proto=proto, mode=mode,
            labels=None, old_limit=old_limit)
        mixed_out, mixed_pair = self._ldc_estimate_transport(
            old_feats=mixed_old, new_feats=mixed_new, proto=proto, mode=mode,
            labels=mixed_labels, old_limit=old_limit)

        current_weight = float(getattr(self.args, 'ldc_hybrid_current_weight', 0.5))
        current_weight = max(0.0, min(1.0, current_weight))
        out = current_weight * current_out + (1.0 - current_weight) * mixed_out
        out = self._ldc_apply_task_residual(out, mixed_pair, mixed_new, labels=mixed_labels, old_limit=old_limit)

        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            out = F.normalize(out, dim=1)
        shrink = float(getattr(self.args, 'ldc_transport_shrink', 1.0))
        if shrink < 1.0:
            out = (1.0 - shrink) * proto + shrink * out
            if int(getattr(self.args, 'ldc_norm_eval', 1)):
                out = F.normalize(out, dim=1)
        valid_cpu = valid.detach().cpu()
        proto_block = self.ldc_proto[:old_limit].clone()
        proto_block[valid_cpu] = out.detach().to(proto_block.device)[valid_cpu]
        self.ldc_proto[:old_limit] = proto_block

    @torch.no_grad()
    def _ldc_estimate_transport(self, old_feats, new_feats, proto, mode, labels, old_limit):
        if mode == 'none':
            return proto, old_feats
        if mode == 'translation':
            delta = (new_feats - old_feats).mean(dim=0, keepdim=True)
            return proto + delta, old_feats + delta
        if mode == 'local_translation':
            X = old_feats
            Y = new_feats
            k = max(1, min(int(getattr(self.args, 'ldc_local_k', 32)), X.size(0)))
            temp = max(float(getattr(self.args, 'ldc_local_temp', 0.05)), 1e-6)
            sim = proto @ X.T
            vals, idx = torch.topk(sim, k=k, dim=1)
            local_delta = (Y - X)[idx]
            weights = torch.softmax(vals / temp, dim=1).unsqueeze(2)
            proto_delta = (weights * local_delta).sum(dim=1)
            return proto + proto_delta, new_feats
        if mode == 'orthogonal':
            X = old_feats
            Y = new_feats
            weights = self._ldc_pair_weights(labels, old_limit=old_limit, n_pairs=X.size(0), device=X.device,
                                             dtype=X.dtype)
            if weights is not None:
                sqrt_w = weights.clamp_min(0).sqrt().unsqueeze(1)
                X_fit = X * sqrt_w
                Y_fit = Y * sqrt_w
            else:
                X_fit, Y_fit = X, Y
            cross = X_fit.T @ Y_fit
            try:
                u, _, vh = torch.linalg.svd(cross, full_matrices=False)
                w = u @ vh
            except RuntimeError:
                w = torch.eye(X.size(1), device=X.device, dtype=X.dtype)
            return proto @ w, old_feats @ w

        use_bias = mode == 'affine'
        X = old_feats
        pair_X = old_feats
        if use_bias:
            X = torch.cat([X, torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)], dim=1)
            pair_X = torch.cat([pair_X, torch.ones(pair_X.size(0), 1, device=pair_X.device,
                                                   dtype=pair_X.dtype)], dim=1)
        Y = new_feats
        weights = self._ldc_pair_weights(labels, old_limit=old_limit, n_pairs=X.size(0), device=X.device,
                                         dtype=X.dtype)
        W = self._ldc_solve_transport(X, Y, weights=weights, use_bias=use_bias)
        robust_iters = max(0, int(getattr(self.args, 'ldc_robust_iters', 0)))
        if robust_iters > 0:
            W = self._ldc_robust_refit(X, Y, W, base_weights=weights, use_bias=use_bias,
                                       robust_iters=robust_iters)
        P = proto
        if use_bias:
            P = torch.cat([P, torch.ones(P.size(0), 1, device=P.device, dtype=P.dtype)], dim=1)
        return P @ W, pair_X @ W

    @torch.no_grad()
    def _ldc_solve_transport(self, X, Y, weights, use_bias):
        if weights is not None:
            weights = weights.to(device=X.device, dtype=X.dtype)
            weights = weights * (float(weights.numel()) / weights.sum().clamp_min(1e-12))
            sqrt_w = weights.clamp_min(0).sqrt().unsqueeze(1)
            X_fit = X * sqrt_w
            Y_fit = Y * sqrt_w
        else:
            X_fit, Y_fit = X, Y

        dim = X_fit.size(1)
        ridge = float(getattr(self.args, 'ldc_ridge', 1e-3))
        eye = torch.eye(dim, device=X_fit.device, dtype=X_fit.dtype)
        if use_bias:
            eye[-1, -1] = 0.0
        lhs = X_fit.T @ X_fit + ridge * eye
        rhs = X_fit.T @ Y_fit
        try:
            W = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            W = torch.linalg.pinv(lhs) @ rhs
        return self._ldc_regularize_transport(W, use_bias=use_bias)

    @torch.no_grad()
    def _ldc_robust_refit(self, X, Y, W, base_weights, use_bias, robust_iters):
        delta_mult = max(float(getattr(self.args, 'ldc_robust_delta_mult', 2.0)), 1e-6)
        if base_weights is not None:
            base_weights = base_weights.to(device=X.device, dtype=X.dtype)
        for _ in range(robust_iters):
            residual = (X @ W - Y).norm(dim=1)
            median = residual.detach().median().clamp_min(1e-6)
            delta = delta_mult * median
            robust_weights = torch.clamp(delta / residual.clamp_min(1e-6), max=1.0)
            if base_weights is not None:
                weights = base_weights * robust_weights
            else:
                weights = robust_weights
            W = self._ldc_solve_transport(X, Y, weights=weights, use_bias=use_bias)
        return W

    @torch.no_grad()
    def _ldc_regularize_transport(self, W, use_bias):
        rho = float(getattr(self.args, 'ldc_spectral_rho', 1.0))
        smin = float(getattr(self.args, 'ldc_spectral_smin', 0.0))
        smax = float(getattr(self.args, 'ldc_spectral_smax', 0.0))
        if rho == 1.0 and smin <= 0.0 and smax <= 0.0:
            return W

        if use_bias:
            linear = W[:-1]
            bias = W[-1:].clone()
        else:
            linear = W
            bias = None

        try:
            u, s, vh = torch.linalg.svd(linear, full_matrices=False)
        except RuntimeError:
            return W

        if rho != 1.0:
            s = 1.0 + rho * (s - 1.0)
        if smin > 0.0 or smax > 0.0:
            lo = smin if smin > 0.0 else None
            hi = smax if smax > 0.0 else None
            s = torch.clamp(s, min=lo, max=hi)
        reg = (u * s.unsqueeze(0)) @ vh

        if bias is None:
            return reg
        return torch.cat([reg, bias], dim=0)

    @torch.no_grad()
    def _ldc_apply_task_residual(self, transported_proto, pair_mapped, new_feats, labels, old_limit):
        strength = float(getattr(self.args, 'ldc_task_residual', 0.0))
        if strength <= 0.0 or labels is None or old_limit <= 0:
            return transported_proto
        labels = labels.to(device=transported_proto.device, dtype=torch.long)
        old_mask = labels < old_limit
        if not torch.any(old_mask):
            return transported_proto

        cpt = int(self.dataset.N_CLASSES_PER_TASK)
        residual = new_feats.to(transported_proto.device) - pair_mapped.to(transported_proto.device)
        out = transported_proto.clone()
        task_ids = torch.unique(labels[old_mask] // cpt)
        for task_id in task_ids:
            task_start = int(task_id.item()) * cpt
            task_end = min(task_start + cpt, old_limit)
            if task_start >= task_end:
                continue
            mask = old_mask & (labels >= task_start) & (labels < task_end)
            if not torch.any(mask):
                continue
            task_delta = residual[mask].mean(dim=0, keepdim=True)
            out[task_start:task_end] = out[task_start:task_end] + strength * task_delta
        return out

    @torch.no_grad()
    def _ldc_pair_weights(self, labels, old_limit, n_pairs, device, dtype):
        mode = str(getattr(self.args, 'ldc_fit_weighting', 'none'))
        if mode == 'none' or labels is None:
            return None
        labels = labels.to(device=device, dtype=torch.long)
        weights = torch.ones(n_pairs, device=device, dtype=dtype)
        if 'source' in mode:
            old_weight = float(getattr(self.args, 'ldc_old_pair_weight', 1.0))
            weights = torch.where(labels < old_limit, weights * old_weight, weights)
        if 'class' in mode:
            uniq, inv, counts = torch.unique(labels, sorted=False, return_inverse=True, return_counts=True)
            class_w = 1.0 / counts.to(dtype=dtype).clamp_min(1)
            weights = weights * class_w[inv]
        return weights * (float(n_pairs) / weights.sum().clamp_min(1e-12))

    @torch.no_grad()
    def _ldc_set_current_prototypes(self, features, labels, start, end):
        dim = features.size(1)
        sums = torch.zeros(end - start, dim, device=features.device, dtype=features.dtype)
        counts = torch.zeros(end - start, device=features.device, dtype=features.dtype)
        local = labels - start
        sums.index_add_(0, local, features)
        counts.index_add_(0, local, torch.ones_like(local, dtype=features.dtype))
        valid = counts > 0
        proto = sums / counts.clamp_min(1).unsqueeze(1)
        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            proto = F.normalize(proto, dim=1)
        valid_cpu = valid.detach().cpu()
        proto_block = self.ldc_proto[start:end].clone()
        count_block = self.ldc_counts[start:end].clone()
        proto_block[valid_cpu] = proto[valid].detach().to(proto_block.device)
        count_block[valid_cpu] = counts[valid].detach().to(count_block.device)
        self.ldc_proto[start:end] = proto_block
        self.ldc_counts[start:end] = count_block

    @torch.no_grad()
    def _ldc_blend_old_buffer_centroids(self, dataset, old_limit):
        blend = float(getattr(self.args, 'ldc_buffer_blend', 0.0))
        if blend <= 0 or old_limit <= 0:
            return

        dim = self.ldc_proto.shape[1]
        sums = torch.zeros(old_limit, dim, device=self.device)
        counts = torch.zeros(old_limit, device=self.device)
        for data in dataset.train_loader:
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device, dtype=torch.long)
            mask = (labels >= 0) & (labels < old_limit)
            if not torch.any(mask):
                continue
            feats = self.net(inputs[mask], returnt='features')
            if int(getattr(self.args, 'ldc_norm_eval', 1)):
                feats = F.normalize(feats, dim=1)
            labs = labels[mask]
            sums.index_add_(0, labs, feats)
            counts.index_add_(0, labs, torch.ones_like(labs, dtype=torch.float32))

        valid = counts > 0
        if not torch.any(valid):
            return
        buf_proto = sums / counts.clamp_min(1).unsqueeze(1)
        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            buf_proto = F.normalize(buf_proto, dim=1)

        old_proto = self.ldc_proto[:old_limit].to(self.device)
        mixed = old_proto.clone()
        mixed[valid] = (1.0 - blend) * old_proto[valid] + blend * buf_proto[valid]
        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            mixed = F.normalize(mixed, dim=1)
        block = self.ldc_proto[:old_limit].clone()
        valid_block = valid.to(block.device)
        block[valid_block] = mixed.detach().to(block.device)[valid_block]
        self.ldc_proto[:old_limit] = block

    @torch.no_grad()
    def _ldc_anchor_prototypes(self, old_limit):
        strength = float(getattr(self.args, 'ldc_proto_anchor', 0.0))
        if strength <= 0.0 or self.etf is None or self.class_num <= 0:
            return
        scope = str(getattr(self.args, 'ldc_proto_anchor_scope', 'old'))
        if scope == 'old':
            start, end = 0, old_limit
        else:
            start, end = 0, self.class_num
        if end <= start:
            return
        valid = self.ldc_counts[start:end] > 0
        if not torch.any(valid):
            return
        proto = self.ldc_proto[start:end].to(self.device)
        etf = self.etf[:, start:end].T.to(device=self.device, dtype=proto.dtype)
        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            proto = F.normalize(proto, dim=1)
            etf = F.normalize(etf, dim=1)
        anchored = (1.0 - strength) * proto + strength * etf
        if int(getattr(self.args, 'ldc_norm_eval', 1)):
            anchored = F.normalize(anchored, dim=1)
        block = self.ldc_proto[start:end].clone()
        valid_cpu = valid.detach().cpu()
        block[valid_cpu] = anchored.detach().to(block.device)[valid_cpu]
        self.ldc_proto[start:end] = block
