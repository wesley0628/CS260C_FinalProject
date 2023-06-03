import argparse
import math
import random
import numpy
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network, IndexedDataset
from torch.utils.data import Dataset, DataLoader


# class IndexedDataset(Dataset):
#     def __init__(self, dataset_name, data_path):
#         _, _, _, _, _, _, dst_train, _, _ = get_dataset(dataset_name, data_path)
#         self.dataset = dst_train
#
#     def __getitem__(self, index):
#         data, target = self.dataset[index]
#         return data, target, index
#
#     def __len__(self):
#         return len(self.dataset)


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


def calculate_el2n(loader, model, TRAIN_NUM, device):
    model.eval()
    el2n_score = torch.zeros(TRAIN_NUM).to(device)
    labels = torch.zeros(TRAIN_NUM, dtype=torch.int)
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            input_var = input.to(device)
            output = model(input_var).to(device)
            normalized_output = nn.Softmax(dim=1)(output)
            curr_el2n = torch.linalg.norm(normalized_output -
                                          nn.functional.one_hot(target,
                                                                num_classes=normalized_output.size(-1)).
                                          to(device), dim=1)
            labels[idx] = target.int()
            el2n_score[idx] = curr_el2n
    return el2n_score.cpu().data.numpy(), labels.cpu().data.numpy()


def validate(val_loader, model, criterion, device, current_epoch, epochs):
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

    print('Epoch:{}/{} * Prec@1 {top1.avg:.3f}'.format(current_epoch, epochs, top1=top1))
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


def calculate_gradient_variance(filter_matrix):
    gradient_variances = np.var(filter_matrix, axis=1)
    gradient_variances = np.transpose(gradient_variances)
    return gradient_variances


def get_class_subset(importance_matrix, labels, class_num, ratio, sample_method):
    current_class_index = np.where(labels == class_num)
    if sample_method == "ranking":
        class_importance_subset = [(index, importance_matrix[index]) for index in current_class_index[0]]
        sorted_class_importance_subset = sorted(class_importance_subset, key=lambda x: x[1], reverse=True)
        class_importance_rank = [t[0] for t in sorted_class_importance_subset]
        selected_index = class_importance_rank[:int(ratio * len(class_importance_subset))]
    elif sample_method == "sampling":
        total_value = sum(importance_matrix[current_class_index])
        class_importance_subset = [(index, importance_matrix[index] / total_value) for index in current_class_index[0]]
        probabilities = [t[1] for t in class_importance_subset]
        class_importance = [t[0] for t in class_importance_subset]
        selected_index = np.random.choice(a=class_importance, size=int(ratio * len(class_importance_subset)),
                                          replace=False, p=probabilities)
        selected_index = selected_index.tolist()
    elif sample_method == "slice":
        class_importance_subset = [(index, importance_matrix[index]) for index in current_class_index[0]]
        from_ = int(0.75 * len(class_importance_subset))
        to_ = int(0.85 * len(class_importance_subset))
        sorted_class_importance_subset = sorted(class_importance_subset, key=lambda x: x[1], reverse=True)
        class_importance_rank = [t[0] for t in sorted_class_importance_subset]
        selected_index = class_importance_rank[from_:to_]
    else:
        exit(1)
    return selected_index


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--filter_method', type=str, default='el2n')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--ratio', type=str, default='0.5')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=str, default='256')
    parser.add_argument('--epochs', type=str, default='30')
    parser.add_argument('--workers', type=str, default='0')
    parser.add_argument('--sample_method', type=str, default='slice')
    args = parser.parse_args()

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    dst_dataset = IndexedDataset(args.dataset, args.data_path)
    _, _, num_classes, _, _, _, _, _, val_loader = get_dataset(args.dataset, args.data_path)
    train_dl = DataLoader(dst_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.workers),
                          pin_memory=True)

    train_criterion = nn.CrossEntropyLoss(reduction='none').to(device)  # (Note)
    val_criterion = nn.CrossEntropyLoss().to(device)
    model = get_model(args.model, args.dataset, args.data_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = int(args.epochs)
    filter_matrix = None
    subset_index = list()
    total_label = None
    all_el2n_score = None

    if args.filter_method == 'el2n':
        epochs = 15
        for i in range(epochs):
            train(train_loader=train_dl, model=model, criterion=train_criterion, optimizer=optimizer, device=device)
            validate(val_loader=val_loader, model=model, criterion=val_criterion, device=device, current_epoch=i + 1,
                     epochs=epochs)
            current_el2n, total_label = calculate_el2n(loader=train_dl, model=model,
                                                       TRAIN_NUM=len(dst_dataset), device=device)
            if all_el2n_score is None:
                all_el2n_score = current_el2n
            else:
                all_el2n_score = np.vstack((all_el2n_score, current_el2n))
    else:
        for i in range(epochs):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(train_loader=train_dl, model=model, criterion=train_criterion, optimizer=optimizer, device=device)
            validate(val_loader=val_loader, model=model, criterion=val_criterion, device=device, current_epoch=i + 1,
                     epochs=epochs)
            pred, label = predictions(loader=train_dl, model=model, TRAIN_NUM=len(dst_dataset), CLASS_NUM=num_classes,
                                      device=device)
            total_label = label
            if i + 1 > 10:
                if args.filter_method == "forgetting":
                    pred_result = (np.argmax(pred, axis=1) == label).astype(int)
                    if filter_matrix is None:
                        filter_matrix = pred_result
                    else:
                        filter_matrix = np.vstack((filter_matrix, pred_result))
                elif args.filter_method == "loss_variance" or args.filter_method == "root_squared_loss":
                    cur_loss = train_criterion(torch.tensor(pred), torch.LongTensor(label))
                    if filter_matrix is None:
                        filter_matrix = np.sqrt(cur_loss)
                    else:
                        filter_matrix = np.vstack((filter_matrix, np.sqrt(cur_loss)))
                elif args.filter_method == "gradient_variance":
                    features = pred - np.eye(num_classes)[label]
                    largest_idx = np.expand_dims(np.argmax(pred, axis=1), axis=1)
                    if filter_matrix is None:
                        filter_matrix = np.take_along_axis(features, largest_idx, axis=1)
                    else:
                        filter_matrix = np.concatenate(
                            (filter_matrix, np.take_along_axis(features, largest_idx, axis=1)),
                            axis=1)
                else:
                    exit(1)

    if args.filter_method == "forgetting":
        importance_list = calculate_forgetting_score(filter_matrix)
    elif args.filter_method == "loss_variance":
        importance_list = np.var(filter_matrix, axis=0)
    elif args.filter_method == "root_squared_loss":
        importance_list = np.average(filter_matrix, axis=0)
    elif args.filter_method == "gradient_variance":
        importance_list = calculate_gradient_variance(filter_matrix)
    elif args.filter_method == "el2n":
        importance_list = all_el2n_score[-1, :]
    else:
        exit(1)

    for i in range(0, num_classes):
        subset_index += get_class_subset(importance_matrix=importance_list, labels=total_label,
                                         class_num=i, ratio=float(args.ratio), sample_method=args.sample_method)
    subset_index = np.array(subset_index)
    indexed_subset = torch.utils.data.Subset(dst_dataset, indices=subset_index)
    torch.save(indexed_subset, 'subset_{}_{}_{}.pth'.format(args.dataset, args.filter_method, args.sample_method))
    print("subset of {} save to file!".format(args.dataset))


if __name__ == '__main__':
    main()
