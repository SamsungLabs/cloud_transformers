# reference https://github.com/zeliu98/CloserLook3D
import torch
from torch import nn
import numpy as np
import time
from datasets.s3dis_closer_utils import AverageMeter, s3dis_metrics, s3dis_metrics_save, sub_s3dis_metrics, s3dis_part_metrics, BatchPointcloudRandomRotate, BatchPointcloudScaleAndJitter
from pathlib import Path
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        loss = F.cross_entropy(logit, target, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config, logger):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = points.size(0)
        # forward
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)

        # print('p', points.shape, 'm', mask.shape, 'f', features.shape)

        pred = model(points, mask, features)

        # print('pred', pred.shape)

        loss = criterion(pred, points_labels, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logger.info(f'Train: [{epoch}/{config.epochs + 1}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
            # logger.info(f'[{cloud_label}]: {input_inds}')
    return loss_meter.avg


def validate(epoch, test_loader, model, criterion, runing_vote_logits, config, logger, num_votes=10, save_path=None):
    """
    One epoch validating
    """
    vote_logits_sum = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]
    vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]
    validation_proj = test_loader.dataset.projections
    validation_labels = test_loader.dataset.clouds_points_labels
    test_smooth = 0.95

    val_proportions = np.zeros(config.num_classes, dtype=np.float32)
    for label_value in range(config.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.clouds_points_labels])

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        RT = BatchPointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                         z_range=config.z_angle_range)
        TS = BatchPointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                           std=config.noise_std, clip=config.noise_clip,
                                           augment_symmetries=config.augment_symmetries)
        for v in range(num_votes):
            test_loader.dataset.epoch = (0 + v) if isinstance(epoch, str) else (epoch + v) % 20
            predictions = []
            targets = []
            for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(test_loader):
                # augment for voting
                if v > 0:
                    points = RT(points)
                    points = TS(points)
                    if config.input_features_dim <= 5:
                        pass
                    elif config.input_features_dim == 6:
                        color = features[:, :3, :]
                        features = torch.cat([color, points.transpose(1, 2).contiguous()], 1)
                    elif config.input_features_dim == 7:
                        color_h = features[:, :4, :]
                        features = torch.cat([color_h, points.transpose(1, 2).contiguous()], 1)
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.input_features_dim} in voting not supported")
                # forward
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                cloud_label = cloud_label.cuda(non_blocking=True)
                input_inds = input_inds.cuda(non_blocking=True)

                pred = model(points, mask, features)
                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(np.bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]
                    runing_vote_logits[c_i][:, inds] = test_smooth * runing_vote_logits[c_i][:, inds] + \
                                                       (1 - test_smooth) * logits
                    predictions += [logits]
                    targets += [test_loader.dataset.sub_clouds_points_labels[c_i][inds]]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % config.print_freq == 0:
                    logger.info(
                        f'Test: [{idx}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            pIoUs, pmIoU = s3dis_part_metrics(config.num_classes, predictions, targets, val_proportions)

            logger.info(f'E{epoch} V{v} * part_mIoU {pmIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * part_msIoU {pIoUs}')

            runsubIoUs, runsubmIoU = sub_s3dis_metrics(config.num_classes, runing_vote_logits,
                                                       test_loader.dataset.sub_clouds_points_labels, val_proportions)
            logger.info(f'E{epoch} V{v} * running sub_mIoU {runsubmIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * running sub_msIoU {runsubIoUs}')

            subIoUs, submIoU = sub_s3dis_metrics(config.num_classes, vote_logits,
                                                 test_loader.dataset.sub_clouds_points_labels, val_proportions)
            logger.info(f'E{epoch} V{v} * sub_mIoU {submIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * sub_msIoU {subIoUs}')

            if save_path is not None:
                IoUs, mIoU = s3dis_metrics_save(config.num_classes, vote_logits, validation_proj, validation_labels,
                                                str(Path(save_path).joinpath(f'preds_{epoch}_{v}.pickle')))
            else:
                IoUs, mIoU = s3dis_metrics(config.num_classes, vote_logits, validation_proj, validation_labels)
            logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')

    return mIoU