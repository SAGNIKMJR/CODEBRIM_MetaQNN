class AverageMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = self.avg = self.sum = self.count = 0

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
    """
    evaluates a model's top k accuracy
    :param output: model output
    :param target: ground-truths/labels
    :param topk: list of integers specifying top-k precisions to be computed
    :return percentage of correct predictions
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
