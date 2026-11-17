"""Numerical invariants of the released scoring and metric implementation."""

import unittest

import torch
import torch.nn.functional as F

from rcr.rcr import (competitive_radius, fit_linear_transport,
                     nc_shrink_prototypes, rcr_scores)
from scripts.evaluate_artifacts import forgetting, score_metrics


class RCRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)

    def test_nc_limit_preserves_etf_decision(self):
        vertices = F.normalize(torch.eye(6) - torch.ones(6, 6) / 6, dim=1)
        features = torch.randn(19, 6)
        scores = rcr_scores(features, vertices, vertices, torch.ones(6), 2)
        self.assertTrue(torch.equal(scores.argmax(1), (features @ vertices.T).argmax(1)))

    def test_inference_does_not_depend_on_other_queries(self):
        vertices = F.normalize(torch.randn(6, 8), dim=1)
        proto = F.normalize(torch.randn(6, 8), dim=1)
        features = torch.randn(19, 8)
        radius = torch.rand(6) + .1
        together = rcr_scores(features, vertices, proto, radius, 2)
        separate = torch.cat([rcr_scores(z[None], vertices, proto, radius, 2) for z in features])
        torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)

    def test_radius_budget_and_shrink_are_finite(self):
        radius = torch.tensor([.1, .2, .7])
        competitive = competitive_radius(radius)
        torch.testing.assert_close(competitive.mean(), radius.mean())
        vertices = F.normalize(torch.randn(3, 8), dim=1)
        proto = nc_shrink_prototypes(vertices, vertices, competitive)
        torch.testing.assert_close(proto, vertices)

    def test_transport_identity(self):
        x = torch.randn(32, 8)
        transport = fit_linear_transport(x, x, ridge=0)
        torch.testing.assert_close(transport, torch.eye(8), atol=1e-5, rtol=1e-5)

    def test_error_partition_and_forgetting(self):
        scores = torch.tensor([[3., 0., 0., 0.], [0., 0., 2., 1.], [0., 0., 0., 3.]])
        metrics = score_metrics(scores, torch.tensor([0, 1, 2]), 2)
        self.assertAlmostEqual(sum(metrics[k] for k in ["class_il", "wrong_task", "same_task_error"]), 100., places=4)
        self.assertAlmostEqual(forgetting([[90.], [80., 85.], [70., 75., 95.]]), 15.)


if __name__ == "__main__":
    unittest.main()
