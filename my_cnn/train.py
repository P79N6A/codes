import os
import sys
import torch
import logging
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve, auc


# https://blog.csdn.net/Dinosoft/article/details/43114935 选模型的标准是auc


def compute_loss(logit, target):
    # weight = torch.tensor([0.1, 0.90])
    weight = torch.tensor([0.1, 0.90])
    # loss = F.cross_entropy(logit, target)
    loss = F.cross_entropy(logit, target, weight)
    return loss


def compute_metric(preds, labels, pos_label=1):
    if len(set(preds)) <= 1:
        return {}
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    auc_score = auc(fpr, tpr)
    acc = (preds == labels).mean()
    num = (preds == labels).sum()
    total = labels.shape[0]
    f1 = f1_score(y_true=labels, y_pred=preds, pos_label=pos_label)
    p, r, f, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, pos_label=pos_label)
    return {
        'auc': round(auc_score, 2),
        "f1": round(f1, 2),
        "acc": "%s(%s/%s)" % (round(acc, 2), num, total),
        'f': np.around(f, decimals=2).tolist(),
        'p1': np.around(p, decimals=2).tolist(),
        'r1': np.around(r, decimals=2).tolist(),
    }


def train(train_iter, dev_iter, model, args, total_steps):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_auc = 0
    last_step = 0
    model.train()
    # all_steps = train_iter.
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            # print('logit vector', logit.size())
            # print('target vector', target.size())

            loss = compute_loss(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                preds = torch.max(logit, 1)[1].view(target.size()).data.numpy()
                labels = target.data.numpy()
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                # accuracy = 100.0 * corrects / batch.batch_size
                detail = compute_metric(preds, labels)
                # sys.stdout.write(
                #     '\rBatch[{}/{}] - loss: {:.6f}  auc:{} detail: {}'.format(steps, total_steps,
                #                                                            round(loss.data.item(), 2),
                #                                                            detail.get('auc'),
                #                                                            detail))
                logging.critical('\rBatch[{}/{}/{}] - loss: {:.2f}  auc:{} detail: {}'.format(steps, total_steps, epoch,
                                                                                              round(loss.data.item(),
                                                                                                    2),
                                                                                              detail.get('auc'),
                                                                                              detail))
            if steps % args.test_interval == 0:
                dev_auc = eval(dev_iter, model, args, steps, total_steps, epoch)
                if dev_auc > best_auc:
                    best_auc = dev_auc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
        save(model, args.save_dir, 'epoch', epoch)
        eval(dev_iter, model, args, steps, total_steps, epoch)


def eval(data_iter, model, args, steps, total_steps, epoch):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = compute_loss(logit, target)
        avg_loss += loss.data.item()
        # corrects += (torch.max(logit, 1)
        #              [1].view(target.size()).data == target.data).sum()
        preds = torch.max(logit, 1)[1].view(target.size()).data.numpy()
        labels = target.data.numpy()
        detail = compute_metric(preds, labels)
        auc = detail.get('auc')
    size = len(data_iter.dataset)
    avg_loss /= size
    # accuracy = 100.0 * corrects / size
    logging.critical('\nEvaluation[{}/{}/{}] - loss: {:.6f}  auc:{} detail:{}\n'.format(steps, total_steps, epoch,
                                                                                        round(avg_loss, 2),
                                                                                        detail.get('auc'),
                                                                                        detail))
    return auc


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0] + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
