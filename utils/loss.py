# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, imgs):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets, imgs)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gi, gj = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                pcls = pi[b,a,gi,gj,9:]
                # Regression
                out = pi[b,a,gi,gj,:8]
                # print(tbox[i])
                # print(out)
                # print(b)
                # print(a)
                # print(self.anchors[i])
                # print(gi, gj)
                # # print(pi[b,a,gi,gj,8])
                # exit()
                lbox += torch.mean((out - tbox[i]) ** 2) # Mean squared error for now
                # pxy = pxy.sigmoid() * 2 - 0.5
                # pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                # pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # iou = iou.detach().clamp(0).type(tobj.dtype)
                # if self.sort_obj_iou:
                #     j = iou.argsort()
                #     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                # if self.gr < 1:
                #     iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gi, gj] = 1  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 8], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size


        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets, imgs):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # (ImageIDX, x1y1x2y2x3y3x4y4 (comma separated), anchorIDX)
        gain = torch.ones(10, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[1:9] = torch.tensor(shape)[[3, 2, 3, 2, 3 ,2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                # width = t[:,:,5] - t[:,:,1]
                # height = t[:,:,2] - t[:,:,4]
                # print(t)
                # print(width)
                # print(torch.cat([width, height], -1))
                # r =  torch.cat([width, height], -1)/ anchors[:, None]  # wh ratio
                # # print(r)
                # j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[j]  # filter
                # # Offsets

                t = t.view(-1, t.shape[-1])
                gx = torch.sum(t[:,1:9:2], dim = -1, keepdim=True) / 4
                gy = torch.sum(t[:,2:9:2], dim = -1, keepdim=True) / 4
                gxy = torch.cat([gx, gy], dim = -1)
                # gxy is basically the center or the trapezium
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                # Offsets are basically for considering grid cells to the top, left, bottom, right as well, based on whether
                # the ground truth box center is lying to the left or right to the grid center
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # print(offsets)
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            # a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # gij = (gxy - offsets).long()
            # gi, gj = gij.T  # grid indices
            b, p1, p2, p3, p4, a = torch.split(t, split_size_or_sections=[1,2,2,2,2,1], dim = -1)
            a = a.long().view(-1) # na * targets.shape[0]
            b = b.long()
            c = None # For now
            b = b.squeeze(-1)
            # Calculate the center of the ground truth trapezium and assign the grid cell which contains the center
            # as the grid cell whose anchors will ultimately fit to the ground truth
            gi, gj = (((p1 + p2 + p3 + p4) / 4) - offsets).long().T
            gj.clamp_(0, shape[2] - 1)
            gi.clamp_(0, shape[3] - 1)
            grid_centers = torch.stack([gi + 0.5, gj + 0.5], dim = 0).T
            anc_boxes = anchors[a]
            tl = grid_centers - anc_boxes / 2
            br = grid_centers + anc_boxes / 2
            bl = torch.stack([(grid_centers[:,0] - anc_boxes[:,0]/2), (grid_centers[:,1] + anc_boxes[:,1]/2)], dim = 1)
            tr = torch.stack([(grid_centers[:,0] + anc_boxes[:,0]/2), (grid_centers[:,1] - anc_boxes[:,1]/2)], dim = 1)
            # for j in range(tl.shape[0]):
            #     # Visualize the anchor boxes first
            #     im = cv2.cvtColor(imgs[0].numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB)
            #     print(im)
            #     pts = np.concatenate([tl[j].numpy(), bl[j].numpy(), br[j].numpy(), tr[j].numpy()], axis = -1)
            #     pt = pts.reshape((-1, 1, 2))
            #     # im = cv2.polylines(im, [np.int32(pt * 8 * (2 ** i))], True, (0,255,0), 2)

            #     im = cv2.rectangle(im, np.int32((bl[j] * 8 * (2 ** i)).numpy()), np.int32((r[j] * 8 * (2 ** i)).numpy()), (0,255,0), 2)
            #     cv2.imshow("yolo", im)
            #     cv2.waitKey(0)
            offsets = torch.cat((p1 - bl, p2 - tl, p3 - tr, p4 - br), dim = -1) # true value of p[i][b,a,gi,gj,:8]
            # Append
            # print(offsets)
            # exit(0)
            indices.append((b, a, gi, gj))  # image, anchor, grid
            tbox.append(offsets)  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch