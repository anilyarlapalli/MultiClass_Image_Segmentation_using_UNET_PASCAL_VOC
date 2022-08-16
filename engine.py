from tqdm import tqdm
import torch
import config
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from d2l import torch as d2l

astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)

def gpu(i=0):
    return torch.device(f'cuda:{i}')

def try_all_gpus():
    gpus_count = torch.cuda.device_count()
    return [gpu(i) for i in range(gpus_count)]

class Accumulator:
    # """For accumulating sums over `n` variables."""
    def __init__(self, n):

        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
 

def accuracy(y_hat, y):
    # """Compute the number of correct predictions."""

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def train_batch(net, X, y, loss, trainer, devices):
    # """Train for a minibatch with mutiple GPUs """
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)

    return train_loss_sum, train_acc_sum

def evaluate_accuracy_gpu(net, data_iter, device=None):
    # """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def train_fn(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices = try_all_gpus()):
    # """Train a model with mutiple GPUs"""

    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    min_acc = 0

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch( net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[3]

        print(f'Epoch: {epoch}, Train loss {train_loss:.3f}, train acc 'f'{train_acc:.3f}, test acc {test_acc:.3f}')
        # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' f'{str(devices)}')

        if min_acc < test_acc:
            min_acc = test_acc
            torch.save({ 'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': trainer.state_dict(),
                'loss': train_loss,}, "weights/best.pt")
