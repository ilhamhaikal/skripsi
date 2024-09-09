import json
import matplotlib.pyplot as plt
import config as conf


def show_history(val_history, train_history, history):
    plt.plot(val_history, color='r', label='Test')
    plt.plot(train_history, color='g', label='Train')
    plt.xlabel("Epoch")
    plt.ylabel(history)
    plt.legend()
    plt.show()


with open(conf.OUT_DIR_HISTORY + 'Accuracy.json', 'r') as openfile:
    accuracy = json.load(openfile)

with open(conf.OUT_DIR_HISTORY + 'Loss.json', 'r') as openfile:
    loss = json.load(openfile)

val_cost_history = loss["val"]
train_cost_history = loss["train"]

val_acc_history = accuracy["val"]
train_acc_history = accuracy["train"]

show_history(val_history=val_cost_history, train_history=train_cost_history, history="Loss")
show_history(val_history=val_acc_history, train_history=train_acc_history, history="Accuracy")
