import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(History, fig_name):
    print('Epoch: ' + str(np.argmin(History.history["val_loss"]) + 1))
    print('Lowest Loss: ' + str(np.min(History.history["val_loss"])))

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();

    plt.savefig('results_Lab5/' + fig_name + '.png')
    plt.close()


def plot_validation_metric(History, fig_name):
    print('Epoch: ' + str(np.argmin(History.history["val_mean_absolute_error"]) + 1))
    print('Lowest error: ' + str(np.min(History.history["val_mean_absolute_error"])))

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve_metrics")
    plt.plot(History.history["mean_absolute_error"], label="mean_absolute_error")
    plt.plot(History.history["val_mean_absolute_error"], label="val_mean_absolute_error")
    plt.plot(np.argmin(History.history["val_mean_absolute_error"]),
             np.min(History.history["val_mean_absolute_error"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.legend();
    plt.savefig('results_Lab5/' + fig_name + '.png')
    plt.close()


def plot_validation_metric_task2(History, fig_name):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve_metrics")
    plt.plot(History.history["accuracy"], label="accuracy")
    plt.plot(History.history["val_accuracy"], label="val_accuracy")
    plt.plot(np.argmax(History.history["val_accuracy"]),
             np.max(History.history["val_accuracy"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.legend();
    plt.savefig('results_Lab5/' + fig_name + '.png')
    plt.close()



def plot_validation_metric_task3(History, fig_name):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve_metrics")
    plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
    plt.plot(History.history["val_precision"], label="val_precision")
    plt.plot(History.history["val_recall"], label="val_recall")

    plt.xlabel("Epochs")
    plt.legend();
    plt.savefig('results_Lab5/' + fig_name + '.png')
    plt.close()
