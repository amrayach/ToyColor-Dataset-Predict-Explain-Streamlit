import torch
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np




def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = metrics.confusion_matrix(labels, pred_labels);
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(2));
    cm.plot(values_format='d', cmap='Blues', ax=ax)

def plot_most_incorrect(incorrect, n_images):

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim = 0)
        #ax.imshow(image.view(5, 5, 3).cpu().numpy(), cmap='bone')
        ax.imshow(image.permute(1, 2, 0), cmap='bone')

        #ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n' \
        #             f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace= 0.5)


def plot_weights(weights, n_weights):

    rows = int(np.sqrt(n_weights))
    cols = int(np.sqrt(n_weights))

    fig = plt.figure(figsize = (20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(weights[i].view(5, 5, 3).cpu().numpy(), cmap = 'bone')
        ax.axis('off')