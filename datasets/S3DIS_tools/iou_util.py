import numpy as np


from .. import s3dis_v2


class AccuracyIOUAccumulator:
    def __init__(self, num_classes=13):
        self.num_classes = num_classes

        self.gt_classes = [0 for _ in range(num_classes)]
        self.positive_classes = [0 for _ in range(num_classes)]
        self.true_positive_classes = [0 for _ in range(num_classes)]

    def add(self, reduced_list):
        for gt_classes, positive_classes, true_positive_classes in reduced_list:
            assert(len(gt_classes) == self.num_classes)
            assert(len(positive_classes) == self.num_classes)
            assert(len(true_positive_classes) == self.num_classes)

            for class_id in range(self.num_classes):
                self.gt_classes[class_id] += gt_classes[class_id]
                self.positive_classes[class_id] += positive_classes[class_id]
                self.true_positive_classes[class_id] += true_positive_classes[class_id]

    def return_metrics(self):
        metrics = dict()

        metrics['overall_acc'] = sum(self.true_positive_classes)/float(sum(self.positive_classes))

        iou_list = []
        for i in range(self.num_classes):
            iou = self.true_positive_classes[i] / float(self.gt_classes[i] + self.positive_classes[i] - self.true_positive_classes[i])
            iou_list.append(iou)

            metrics['iou_{}'.format(s3dis_v2.class_order[i])] = iou

        metrics['mean_iou'] = sum(iou_list) / self.num_classes

        return metrics


def batch_add(pred_probs, gt_lables):
    num_classes = pred_probs.shape[1]

    gt_classes = [0 for _ in range(num_classes)]
    positive_classes = [0 for _ in range(num_classes)]
    true_positive_classes = [0 for _ in range(num_classes)]

    assert(pred_probs.shape[0] == gt_lables.shape[0])

    # num points
    assert(pred_probs.shape[2] == gt_lables.shape[1])

    pred = np.argmax(pred_probs, axis=1)

    assert(pred.shape == gt_lables.shape)
    for batch_id in range(pred.shape[0]):
        for point_id in range(pred.shape[1]):
            gt_l = gt_lables[batch_id, point_id]
            pred_l = pred[batch_id, point_id]

            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1

            true_positive_classes[gt_l] += int(gt_l == pred_l)

    return gt_classes, positive_classes, true_positive_classes
