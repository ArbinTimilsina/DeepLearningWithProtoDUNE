import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from matplotlib.colors import LogNorm

def plot_weights_median(weights, ranges, class_names, plot_name):
    """
    Plot weights for each class based on the frequencies of the samples and display median.
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(18,5), facecolor='w')
    for index in range(len(class_names)):
        ax = axes[index]
        ax.hist(weights[index], 50, range=ranges[index], color='green', alpha=0.75)
        ax.set_title(class_names[index], fontsize=20, fontname='DejaVu Sans',fontweight='bold')
        ax.set_xlabel("Weight", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
        ax.set_ylabel("Count", fontsize=15, fontname='DejaVu Sans',fontweight='bold')

        _, max = ax.get_ylim()
        median = np.median(weights[index])
        ax.axvline(median, color='k', linestyle='dashed', linewidth=2)
        ax.text(median + median/50, max - max/5, 'Median: {:.1f}'.format(median),
                fontsize=12, fontweight=1000, color='k')

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)

def get_feature_image(feature_image, fig, ax, title):
    """
    Setup for feature image.
    """
    c = ax.imshow(feature_image, cmap='winter_r',interpolation='none', origin='lower',
                  norm=LogNorm(vmin=1.0, vmax=abs(feature_image).max()))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Global wire no.", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax.set_ylabel("TDC", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax.set_title(title, fontsize=20,fontname='DejaVu Sans',fontweight='bold')

def get_label_image(label_image, class_names, fig, ax, title):
    """
    Setup for label image.
    """
    minimum = 0.0
    maximum = len(class_names)
    cmap = plt.get_cmap('gist_heat_r', maximum-minimum)
    c= ax.imshow(label_image,cmap=cmap,interpolation='none', origin='lower',
                   vmin=minimum, vmax=maximum-1)
    cb = fig.colorbar(c, ax=ax)
    cb.set_ticks([x for x in range(len(class_names))])
    cb.set_ticklabels(class_names)

    ax.set_xlabel("Global wire no.", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax.set_ylabel("TDC", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax.set_title(title, fontsize=20,fontname='DejaVu Sans',fontweight='bold')

def plot_feature_label(feature_image, label_image,
                       feature_title, label_tile,
                       class_names, plot_name):
    """
    Plot feature and label side by side.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,7), facecolor='w')

    get_feature_image(feature_image, fig, ax0, feature_title)
    get_label_image(label_image, class_names, fig, ax1, label_tile)

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)

def plot_categories(feature_image, label_image, class_names, plot_name):
    """
    Plot categories side by side.
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(25,7), facecolor='w')
    for index in range(len(class_names)):
        ax = axes[index]
        mask = (label_image == index)
        c = ax.imshow(feature_image*mask, cmap='winter_r',interpolation='none', origin='lower',
                  norm=LogNorm(vmin=1.0, vmax=np.max(feature_image)))
        ax.set_xlabel("Global wire no.", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
        ax.set_ylabel("TDC", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
        ax.set_title(class_names[index], fontsize=20,fontname='DejaVu Sans',fontweight='bold')

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)

def plot_feature_label_prediction(feature_image, label_image, prediction_image,
                                  feature_title, label_title, prediction_title,
                                  class_names, plot_name):
    """
    Plot feature, label, and prediction side by side.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20,5), facecolor='w')

    # Just pick first channel
    feature_image = feature_image[:, :, 0]

    # Featured is scaled to be between 0 and 1
    c = ax0.imshow(feature_image, cmap='winter_r',interpolation='none', origin='lower',
                   norm=LogNorm(vmin=1E-3, vmax=1.0))
    fig.colorbar(c, ax=ax0)
    ax0.set_xlabel("Global wire no.", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax0.set_ylabel("TDC", fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    ax0.set_title(feature_title, fontsize=20,fontname='DejaVu Sans',fontweight='bold')

    get_label_image(label_image, class_names, fig, ax1, label_title)
    get_label_image(prediction_image, class_names, fig, ax2, prediction_title)

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)

def plot_history(history, quantity, plot_title, y_label, plot_name):
    fig, axes = plt.subplots(1, 1, figsize=(15,10), facecolor='w')
    axes.plot(history.history[quantity.lower()])
    axes.plot(history.history['val_{}'.format(quantity.lower())])
    axes.set_title(plot_title, fontsize=20,fontname='DejaVu Sans',fontweight='bold')
    axes.set_ylabel(y_label, fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    axes.set_xlabel('Epoch', fontsize=15, fontname='DejaVu Sans',fontweight='bold')
    axes.legend(['Training', 'Validation'], loc='upper left')

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)
