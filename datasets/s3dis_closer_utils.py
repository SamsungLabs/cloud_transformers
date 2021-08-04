import numpy as np
import torch
import torch.distributed as dist
# reference https://github.com/zeliu98/CloserLook3D
from sklearn.metrics import confusion_matrix
import pickle

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std = std
        self.clip = clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
        symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
        xyz1 *= symmetries
        xyz2 = np.clip(np.random.normal(scale=self.std, size=[pc.shape[0], 3]), a_min=-self.clip, a_max=self.clip)
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc

class PointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class BatchPointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std, self.clip = std, clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        bsize = pc.size()[0]
        npoint = pc.size()[1]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
            symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
            xyz1 *= symmetries
            xyz2 = np.clip(np.random.normal(scale=self.std, size=[npoint, 3]), a_max=self.clip, a_min=-self.clip)

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, pc):
        bsize = pc.size()[0]
        normals = pc.size()[2] > 3
        for i in range(bsize):
            angles_ = self._get_angles()
            Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx).to(pc.device)

            if not normals:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
            else:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
                pc[i, :, 3:] = torch.matmul(pc[i, :, 3:], rotation_matrix.t())
        return pc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def partnet_metrics(num_classes, num_parts, objects, preds, targets):
    """
    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]
    Returns:
    """
    shape_iou_tot = [0.0] * num_classes
    shape_iou_cnt = [0] * num_classes
    part_intersect = [np.zeros((num_parts[o_l]), dtype=np.float32) for o_l in range(num_classes)]
    part_union = [np.zeros((num_parts[o_l]), dtype=np.float32) + 1e-6 for o_l in range(num_classes)]

    for obj, cur_pred, cur_gt in zip(objects, preds, targets):
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred[1:, :], axis=0) + 1
        cur_pred[cur_gt == 0] = 0
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for j in range(1, cur_num_parts):
            cur_gt_mask = (cur_gt == j)
            cur_pred_mask = (cur_pred == j)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[obj][j] += intersect
                part_union[obj][j] += union
        if cur_shape_iou_cnt > 0:
            cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
            shape_iou_tot[obj] += cur_shape_miou
            shape_iou_cnt[obj] += 1

    msIoU = [shape_iou_tot[o_l] / shape_iou_cnt[o_l] for o_l in range(num_classes)]
    part_iou = [np.divide(part_intersect[o_l][1:], part_union[o_l][1:]) for o_l in range(num_classes)]
    mpIoU = [np.mean(part_iou[o_l]) for o_l in range(num_classes)]

    # Print instance mean
    mmsIoU = np.mean(np.array(msIoU))
    mmpIoU = np.mean(mpIoU)

    return msIoU, mpIoU, mmsIoU, mmpIoU


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def s3dis_metrics(num_classes, vote_logits, validation_proj, validation_labels):
    Confs = []
    for logits, proj, targets in zip(vote_logits, validation_proj, validation_labels):
        preds = np.argmax(logits[:, proj], axis=0).astype(np.int32)
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0)

    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)
    return IoUs, mIoU

def s3dis_metrics_save(num_classes, vote_logits, validation_proj, validation_labels, path):
    Confs = []
    all_preds = []
    all_targets = []
    for logits, proj, targets in zip(vote_logits, validation_proj, validation_labels):
        preds = np.argmax(logits[:, proj], axis=0).astype(np.int32)
        all_preds.append(preds)
        all_targets.append(targets)
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0)
    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)

    save_dict = {'all_preds': all_preds,
                 'all_targets': all_targets,
                 'vote_logits': vote_logits,
                 'validation_proj': validation_proj,
                 'validation_labels': validation_labels,
                 'C': C,
                 'IoUs': IoUs,
                 'mIoU': mIoU}

    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)

    return IoUs, mIoU

def sub_s3dis_metrics(num_classes, validation_logits, validation_labels, val_proportions):
    Confs = []
    for logits, targets in zip(validation_logits, validation_labels):
        preds = np.argmax(logits, axis=0).astype(np.int32)
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
    # Rescale with the right number of point per class
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)

    return IoUs, mIoU


def s3dis_part_metrics(num_classes, predictions, targets, val_proportions):
    # Confusions for subparts of validation set
    Confs = np.zeros((len(predictions), num_classes, num_classes), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):
        # Predicted labels
        preds = np.argmax(probs, axis=0)
        # Confusions
        Confs[i, :, :] = confusion_matrix(truth, preds, np.arange(num_classes))
    # Sum all confusions
    C = np.sum(Confs, axis=0).astype(np.float32)
    # Balance with real validation proportions
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    # Objects IoU
    IoUs = IoU_from_confusions(C)
    # Print instance mean
    mIoU = np.mean(IoUs)
    return IoUs, mIoU


def shapenetpart_metrics(num_classes, num_parts, objects, preds, targets, masks):
    """
    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]
        masks: [(num_points)]
    """
    total_correct = 0.0
    total_seen = 0.0
    Confs = []
    for obj, cur_pred, cur_gt, cur_mask in zip(objects, preds, targets, masks):
        obj = int(obj)
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred, axis=0)
        cur_pred = cur_pred[cur_mask]
        cur_gt = cur_gt[cur_mask]
        correct = np.sum(cur_pred == cur_gt)
        total_correct += correct
        total_seen += cur_pred.shape[0]
        parts = [j for j in range(cur_num_parts)]
        Confs += [confusion_matrix(cur_gt, cur_pred, labels=parts)]

    Confs = np.array(Confs)
    obj_mIoUs = []
    objects = np.asarray(objects)
    for l in range(num_classes):
        obj_inds = np.where(objects == l)[0]
        obj_confs = np.stack(Confs[obj_inds])
        obj_IoUs = IoU_from_confusions(obj_confs)
        obj_mIoUs += [np.mean(obj_IoUs, axis=-1)]

    objs_average = [np.mean(mIoUs) for mIoUs in obj_mIoUs]
    instance_average = np.mean(np.hstack(obj_mIoUs))
    class_average = np.mean(objs_average)
    acc = total_correct / total_seen

    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')

    s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
    for AmIoU in objs_average:
        s += '{:4.1f} '.format(100 * AmIoU)
    print(s + '\n')
    return acc, objs_average, class_average, instance_average
