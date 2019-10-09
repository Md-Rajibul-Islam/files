# This is a function file for both of the single plot and subplot
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(History, fig_name):
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

    plt.savefig('results/Lab2/' + fig_name + '.png')
    plt.close()


def plot_learning_curve_subplot(History, fig_name):
#    % matplotlib inline
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)

    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();

    plt.subplot(1, 2, 2)
    plt.title("Learning curve")
    plt.plot(History.history["binary_accuracy"], label="accuracy")
    plt.plot(History.history["val_binary_accuracy"], label="val_accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend();

    plt.savefig('results/Lab2/' + fig_name + '.png')
    plt.close()


def subplot_value_accuracy(History, fig_name):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)

    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();

    plt.subplot(1, 2, 2)
    plt.title("Learning curve")
    plt.plot(History.history["accuracy"], label="accuracy")
    plt.plot(History.history["val_accuracy"], label="val_accuracy")
    plt.plot(np.argmin(History.history["val_accuracy"]),
             np.min(History.history["val_accuracy"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend();

    plt.savefig('results/Lab2/' + fig_name + '.png')
    plt.close()