import itertools

from typing import Dict, List

import numpy as np
import torch

from scipy.ndimage.morphology import binary_dilation
from skimage import measure
from torch import nn

import masters.core.components.malis as m


class LossCollection(nn.Module):
    def __init__(
        self,
        loss_functions: List[nn.Module],
        loss_weights: List[float] | None = None,
    ):
        super().__init__()
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self._current_losses = None

    def forward(self, pred, target) -> Dict[str, torch.Tensor]:
        losses = {}
        for loss_function, loss_weight in zip(self.loss_functions, self.loss_weights):
            loss = loss_function(pred, target)
            loss_name = loss_function.__class__.__name__
            losses[loss_name] = loss * loss_weight
        return losses


class MSEConnLossMultiCube(nn.Module):
    def __init__(
        self,
        alpha: float = 0.0001,
        beta: float = 0.1,
        dmax: int = 20,
        costs_n_cap: float = 10.0,
        costs_p_cap: float = 3.0,
        window: int = 64,
    ):
        super().__init__()
        self.alpha = alpha
        self.conn = ConnLossMultiCube(beta, dmax, costs_n_cap, costs_p_cap, window)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss1 = self.mse(pred, target)
        loss2 = self.alpha * self.conn(pred, target)
        return loss1, loss2


class ConnLossMultiCube(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        dmax: int = 20,
        costs_n_cap: float = 10.0,
        costs_p_cap: float = 3.0,
        window_3d: int = 32,
        window_2d: int = 32,
    ):
        super().__init__()
        self.malis = MALISWindowPossLoss(beta, dmax, costs_n_cap, costs_p_cap, window_2d)
        self.window_3d = window_3d

    def forward(self, pred, target):
        B, C, H, W, D = pred.shape

        losses = []
        window = self.window_3d
        for k, j, i in itertools.product(range(H // window), range(W // window), range(D // window)):
            pred_part = pred[
                :, :, k * window : (k + 1) * window, j * window : (j + 1) * window, i * window : (i + 1) * window
            ]
            target_part = target[
                :, :, k * window : (k + 1) * window, j * window : (j + 1) * window, i * window : (i + 1) * window
            ]
            projection1, _ = pred_part.min(2)
            projection1 = projection1.reshape(
                pred_part.size(0), pred_part.size(1), pred_part.size(3), pred_part.size(4)
            )
            projection2, _ = pred_part.min(3)
            projection2 = projection2.reshape(
                pred_part.size(0), pred_part.size(1), pred_part.size(2), pred_part.size(4)
            )
            projection3, _ = pred_part.min(4)
            projection3 = projection3.reshape(
                pred_part.size(0), pred_part.size(1), pred_part.size(2), pred_part.size(3)
            )

            l1, _ = target_part.min(2)
            l1 = l1.reshape(target_part.size(0), target_part.size(1), target_part.size(3), target_part.size(4))
            l2, _ = target_part.min(3)
            l2 = l2.reshape(target_part.size(0), target_part.size(1), target_part.size(2), target_part.size(4))
            l3, _ = target_part.min(4)
            l3 = l3.reshape(target_part.size(0), target_part.size(1), target_part.size(2), target_part.size(3))
            loss1 = self.malis(projection1, l1)
            loss2 = self.malis(projection2, l2)
            loss3 = self.malis(projection3, l3)
            losses.append(loss1 + loss2 + loss3)
        return sum(losses)


class MSEConnLossProj(nn.Module):
    def __init__(
        self,
        alpha: float = 0.0001,
        beta: float = 0.1,
        dmax: int = 20,
        costs_n_cap: float = 10.0,
        costs_p_cap: float = 3.0,
        window: int = 64,
    ):
        super().__init__()
        self.alpha = alpha
        self.malis = MALISWindowPossLoss(beta, dmax, costs_n_cap, costs_p_cap, window)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 0- batch, 1- channel, 2- height, 3- width, 4- depth
        projection, _ = pred.min(2)
        projection = projection.reshape(pred.size(0), pred.size(1), pred.size(3), pred.size(4))

        l1, _ = target.min(2)
        l1 = l1.reshape(target.size(0), target.size(1), target.size(3), target.size(4))

        loss1 = self.mse(pred, target)
        loss2 = self.alpha * self.malis(projection, l1)
        return loss1, loss2


class MALISWindowPossLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        dmax: int = 20,
        costs_n_cap: float = 10.0,
        costs_p_cap: float = 3.0,
        window: int | None = 64,
    ):
        super().__init__()
        self.dmax = dmax
        self.costs_n_cap = costs_n_cap
        self.costs_p_cap = costs_p_cap
        self.beta = beta
        self.window = window

    def forward(self, pred, target):
        pred_np_full = pred.cpu().detach().numpy()
        target_np_full = target.cpu().detach().numpy()
        B, C, H, W = pred_np_full.shape

        weights_n = np.zeros(pred_np_full.shape, dtype=np.float32)
        weights_p = np.zeros(pred_np_full.shape, dtype=np.float32)

        window = self.window if self.window else W
        for k in range(H // window):
            for j in range(W // window):
                pred_np = pred_np_full[:, :, k * window : (k + 1) * window, j * window : (j + 1) * window]
                target_np = target_np_full[:, :, k * window : (k + 1) * window, j * window : (j + 1) * window]

                nodes_indexes = np.arange(window * window).reshape(window, window)
                nodes_indexes_h = np.vstack([nodes_indexes[:, :-1].ravel(), nodes_indexes[:, 1:].ravel()]).tolist()
                nodes_indexes_v = np.vstack([nodes_indexes[:-1, :].ravel(), nodes_indexes[1:, :].ravel()]).tolist()
                nodes_indexes = np.hstack([nodes_indexes_h, nodes_indexes_v])
                nodes_indexes = np.uint64(nodes_indexes)

                costs_h = (pred_np[:, :, :, :-1] + pred_np[:, :, :, 1:]).reshape(B, -1)
                costs_v = (pred_np[:, :, :-1, :] + pred_np[:, :, 1:, :]).reshape(B, -1)
                costs = np.hstack([costs_h, costs_v])
                costs = np.float32(costs)

                gtcosts_h = (target_np[:, :, :, :-1] + target_np[:, :, :, 1:]).reshape(B, -1)
                gtcosts_v = (target_np[:, :, :-1, :] + target_np[:, :, 1:, :]).reshape(B, -1)
                gtcosts = np.hstack([gtcosts_h, gtcosts_v])
                gtcosts = np.float32(gtcosts)

                costs_n = costs.copy()
                costs_p = costs.copy()

                costs_n[gtcosts > self.costs_n_cap] = self.costs_n_cap
                costs_p[gtcosts < self.costs_p_cap] = 0
                gtcosts[gtcosts > self.costs_n_cap] = self.costs_n_cap

                for i in range(len(pred_np)):
                    sg_gt = measure.label(binary_dilation((target_np[i, 0] == 0), iterations=5) == 0)

                    edge_weights_n = m.malis_loss_weights(
                        sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], nodes_indexes[1], costs_n[i], 0
                    )

                    edge_weights_p = m.malis_loss_weights(
                        sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], nodes_indexes[1], costs_p[i], 1
                    )

                    num_pairs_n = np.sum(edge_weights_n)
                    if num_pairs_n > 0:
                        edge_weights_n = edge_weights_n / num_pairs_n

                    num_pairs_p = np.sum(edge_weights_p)
                    if num_pairs_p > 0:
                        edge_weights_p = edge_weights_p / num_pairs_p

                    edge_weights_n[gtcosts[i] >= self.costs_p_cap] = 0
                    edge_weights_p[gtcosts[i] < self.costs_n_cap] = 0

                    malis_w = edge_weights_n.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window, window - 1), malis_w_v.reshape(window - 1, window)

                    nodes_weights = np.zeros((window, window), np.float32)
                    nodes_weights[:, :-1] += malis_w_h
                    nodes_weights[:, 1:] += malis_w_h
                    nodes_weights[:-1, :] += malis_w_v
                    nodes_weights[1:, :] += malis_w_v

                    weights_n[i, 0, k * window : (k + 1) * window, j * window : (j + 1) * window] = nodes_weights

                    malis_w = edge_weights_p.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window, window - 1), malis_w_v.reshape(window - 1, window)

                    nodes_weights = np.zeros((window, window), np.float32)
                    nodes_weights[:, :-1] += malis_w_h
                    nodes_weights[:, 1:] += malis_w_h
                    nodes_weights[:-1, :] += malis_w_v
                    nodes_weights[1:, :] += malis_w_v

                    weights_p[i, 0, k * window : (k + 1) * window, j * window : (j + 1) * window] = nodes_weights

        loss_n = (pred).pow(2)
        loss_p = (self.dmax - pred).pow(2)
        loss = (
            loss_n * torch.Tensor(weights_n.astype(np.float32)).cuda()
            + self.beta * loss_p * torch.Tensor(weights_p.astype(np.float32)).cuda()
        )

        return loss.sum()
