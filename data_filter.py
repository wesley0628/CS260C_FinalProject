import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network
from torch.utils.data import Dataset, DataLoader


class IndexedDataset(Dataset):
    def __init__(self, dataset_name, data_path):
        _, _, _, _, _, _, dst_train, _, _ = get_dataset(dataset_name, data_path)
        self.dataset = dst_train

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def get_model(model_name, dataset, data_path):
    channel, im_size, num_classes, _, _, _, _, _, _ = get_dataset(dataset, data_path)
    return get_network(model_name, channel, num_classes, im_size)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
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
    """Computes the precision@k for the specified values of k"""
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


def predictions(loader, model, TRAIN_NUM, CLASS_NUM, device):
    """
    Get predictions
    """

    # switch to evaluate mode
    model.eval()

    preds = torch.zeros(TRAIN_NUM, CLASS_NUM).to(device)
    labels = torch.zeros(TRAIN_NUM, dtype=torch.int)
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            input_var = input.to(device)
            preds[idx, :] = nn.Softmax(dim=1)(model(input_var))
            labels[idx] = target.int()

    return preds.cpu().data.numpy(), labels.cpu().data.numpy()


def train(train_loader, model, criterion, optimizer, device):
    """
        Run one train epoch
    """

    # switch to train mode
    model.train()

    for i, (input, target, idx) in enumerate(train_loader):
        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss = loss.mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, device):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


def calculate_forgetting_score(filter_matrix):
    num_rows, num_cols = filter_matrix.shape
    index_changes = np.zeros(num_cols, dtype=int)

    for i in range(1, num_rows):
        current_row = filter_matrix[i]
        previous_row = filter_matrix[i - 1]
        row_changes = np.abs(current_row - previous_row)
        index_changes += row_changes
    return index_changes


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--filter_method', type=str, default='forgetting')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--ratio', type=str, default='0.5')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=str, default='256')
    parser.add_argument('--epochs', type=str, default='3')
    parser.add_argument('--workers', type=str, default='0')
    args = parser.parse_args()

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    dst_dataset = IndexedDataset(args.dataset, args.data_path)
    _, _, num_classes, _, _, _, _, _, val_loader = get_dataset(args.dataset, args.data_path)
    train_dl = DataLoader(dst_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.workers),
                          pin_memory=True)

    train_criterion = nn.CrossEntropyLoss(reduction='none').to(device)  # (Note)
    val_criterion = nn.CrossEntropyLoss().to(device)
    model = get_model(args.model, args.dataset, args.data_path)
    epochs = int(args.epochs)
    filter_matrix = None
    importance_list = None

    for i in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(train_loader=train_dl, model=model, criterion=train_criterion, optimizer=optimizer, device=device)
        validate(val_loader=val_loader, model=model, criterion=val_criterion, device=device)
        pred, label = predictions(loader=train_dl, model=model, TRAIN_NUM=len(dst_dataset), CLASS_NUM=num_classes, device=device)
        if args.filter_method == "forgetting":
            pred_result = (np.argmax(pred, axis=1) == label).astype(int)
            if filter_matrix is None:
                filter_matrix = pred_result
            else:
                filter_matrix = np.vstack((filter_matrix, pred_result))
        elif args.filter_method == "variance":
            pass
        elif args.filter_method == "loss":
            pass
        else:
            exit(1)
    if args.filter_method == "forgetting":
        importance_list = calculate_forgetting_score(filter_matrix)
    elif args.filter_method == "variance":
        pass
    elif args.filter_method == "loss":
        pass
    else:
        exit(1)

    subset_index = (-importance_list).argsort()[:int(float(args.ratio) * len(dst_dataset))]
    indexed_subset = torch.utils.data.Subset(dst_dataset, indices=subset_index)
    torch.save(indexed_subset, 'subset_{}.pth'.format(args.dataset))



if __name__ == '__main__':
    main()
