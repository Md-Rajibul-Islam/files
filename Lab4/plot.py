#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(History, fig_name):
    print('Epoch: '+ str(np.argmin(History.history["val_loss"])+1))
    print('Lowest Loss: '+ str(np.min(History.history["val_loss"])))
    #%matplotlib inline
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('results/Lab4/' + fig_name + '.png')
    plt.close()
    
def plot_validation_metric(History, fig_name):
    print('Epoch: '+ str(np.argmax(History.history["val_dice_coef"])+1))
    print('Highest accuracy: '+ str(np.max(History.history["val_dice_coef"])))
    #%matplotlib inline
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve_metrics")
    plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
    plt.plot(History.history["val_precision"], label="val_precision")
    plt.plot(History.history["val_recall"], label="val_recall")
    plt.plot( np.argmax(History.history["val_dice_coef"]),
             np.max(History.history["val_dice_coef"]),
             marker="x", color="r", label="best model")
    
    plt.xlabel("Epochs")
    plt.legend();
    plt.savefig('results/Lab4/' + fig_name + '.png')
    plt.close()
