from utils import *
from modelCRNN.crnnNet import CRNN
from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import string
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import config as conf
import json


def show_history(val_history, train_history, history):
    plt.plot(val_history, color='r', label='Test')
    plt.plot(train_history, color='g', label='Train')
    plt.xlabel("Epoch")
    plt.ylabel(history)
    plt.legend()
    plt.show()


def save_history(val_history, train_history, history):
    data = {
        "val": val_history,
        "train": train_history
    }

    json_object = json.dumps(data, indent=2)
    name_file = conf.OUT_DIR_HISTORY + history + ".json"
    with open(name_file, "w") as outfile:
        outfile.write(json_object)

# def train(net, data_loader, crit, opt):
#     net.train()
#     n_loss = 0
#     n_correct = 0
#     total_samples = 0        # Variabel total_samples didefinisikan di sini dan diisi dengan nilai awal 0.
#
#     loader = tqdm(data_loader, total=len(data_loader), desc="Train")
#     for data in loader:
#         images, targets = data
#         batch_size = images.size(0)
#         text, length = labelConverter.encode(targets)
#
#         preds = net(images)
#         preds_size = torch.IntTensor([preds.size(0)] * batch_size)
#         loss = crit(preds, text, preds_size, length)
#
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#
#         n_loss += loss.item()
#
#         _, preds = preds.max(2)
#         preds = preds.transpose(1, 0).contiguous().view(-1)
#         text_preds = labelConverter.decode(preds.data, preds_size.data, raw=False)
#
#         # Menghitung akurasi di dalam loop iterasi
#         for pred, target in zip(text_preds, targets):
#             if pred == target:
#                 n_correct += 1
#
#         # Mengupdate total_samples dengan jumlah sampel dalam batch saat ini
#         total_samples += batch_size
#
#     loss = n_loss / len(data_loader)
#     acc = n_correct / total_samples    # Menggunakan total_samples untuk menghitung akurasi
#
#     return loss, acc

def train(data_loader):
    crnnNet.train()
    n_loss = n_correct = n_total = 0
    loader = tqdm(data_loader, total=len(data_loader), desc="Train")
    for data in loader:
        images, targets = data
        batch_size = images.size(0)
        text, length = labelConverter.encode(targets)

        preds = crnnNet(images)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, text, preds_size, length)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n_loss += loss

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text_preds = labelConverter.decode(preds.data, preds_size.data, raw=False)

        for pred, target in zip(text_preds, targets):
            if pred == target:
                n_correct += 1
            n_total += 1

    loss = float(n_loss / len(data_loader))
    # acc = float(n_correct / (len(data_loader) * conf.BATCH_SIZE))
    acc = float(n_correct / n_total)

    return loss, acc


def test(data_loader):
    crnnNet.eval()             # Menempatkan model ke mode evaluasi (tidak ada dropout atau batch normalization)
    n_correct = n_loss = n_total = 0

    loader = tqdm(data_loader, total=len(data_loader), desc="Eval")
    with torch.no_grad():            # Menyatakan bahwa tidak ada perhitungan gradien pada mode evaluasi
        for data in loader:
            images, targets = data
            batch_size = images.size(0)
            text, length = labelConverter.encode(targets)

            preds = crnnNet(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            n_loss += loss

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            text_preds = labelConverter.decode(preds.data, preds_size.data, raw=False)

            for pred, target in zip(text_preds, targets):
                if pred == target:
                    n_correct += 1
                n_total += 1

    # Menampilkan beberapa prediksi, target, dan hasil prediksi sebenarnya (ground truth)
    raw_preds = labelConverter.decode(preds.data, preds_size.data, raw=True)[:10]
    for raw_pred, pred, gt in zip(raw_preds, text_preds, targets):
        print(f'%s => %s, target: %s' % (raw_pred, pred, gt))

    # Menghitung dan mengembalikan loss dan akurasi pada data uji
    loss = float(n_loss / len(data_loader))
    # acc = float(n_correct / (len(data_loader) * conf.BATCH_SIZE))       # * conf.BATCH_SIZE
    acc = float(n_correct / n_total)

    return loss, acc


if __name__ == '__main__':
    root = '90kDICT32px'
    size = (conf.IMG_W, conf.IMG_H)
    classes = conf.NUMBER + conf.ALPHABET
    nclass = len(classes) + 1

    train_dataset = synthDataset(root, annotation_file="annotation_train.txt", n_file=conf.NUM_TRAIN, transform=resizeNormalize(size=size))
    train_loader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=0)

    test_dataset = synthDataset(root=root, annotation_file="annotation_val.txt", n_file=conf.NUM_VAL, transform=resizeNormalize(size=size))
    test_loader = DataLoader(test_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=0)

    labelConverter = strLabelConverter(classes)
    criterion = CTCLoss()
    crnnNet = CRNN(nclass)
    optimizer = optim.Adam(crnnNet.parameters(), lr=conf.LEARNING_RATE)

    epochs = conf.EPOCHS
    train_cost_history = []
    train_acc_history = []
    val_cost_history = []
    val_acc_history = []

    for epoch in range(epochs):
        print("EPOCH ", epoch + 1, "/", epochs)
        train_cost, train_acc = train(train_loader)
        val_cost, val_acc = test(test_loader)

        train_cost_history.append(train_cost)
        train_acc_history.append(train_acc)
        val_cost_history.append(val_cost)
        val_acc_history.append(val_acc)

        torch.save(crnnNet.state_dict(),
                   '{0}/modelCRNN_{1}_{2}.pth'
                   .format(conf.OUT_DIR, epoch, val_acc))

        print('Train loss: %f, Train Accuracy: %f' % (train_cost, train_acc))
        print('Val loss: %f, Val Accuracy: %f' % (val_cost, val_acc))

    save_history(val_history=val_cost_history, train_history=train_cost_history, history="Loss")
    save_history(val_history=val_acc_history, train_history=train_acc_history, history="Accuracy")

    show_history(val_history=val_cost_history, train_history=train_cost_history, history="Loss")
    show_history(val_history=val_acc_history, train_history=train_acc_history, history="Accuracy")
