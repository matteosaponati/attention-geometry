import numpy as np
import matplotlib.pyplot as plt
from .funs import count_outliers

def symmetry_score_boxplot(models):

    SList = [models[key][-1].flatten() for key in list(models.keys())]
    names = list(models.keys())

    plt.figure(figsize=(5,4))

    ## BERT models
    plt.boxplot(SList)

    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.ylabel('symmetry scores')
    plt.xticks(np.arange(1,len(names)+1),names)
    plt.xticks(rotation=50)
    plt.ylim(0,1)
    plt.axhline(y=.5,color='k',linestyle='dashed')

    return

def symmetry_score_scatter(models):

    SList = [models[key][-1].flatten() for key in list(models.keys())]
    names = list(models.keys())

    plt.figure(figsize=(4,4))

    for i, model in enumerate(SList):
        
        plt.scatter(i*np.ones(SList[i].size),(SList[i].flatten()),color='purple',alpha=.1)
        plt.scatter(i,(SList[i].flatten()).mean(),color='purple')

    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.ylabel('symmetry scores')
    plt.xticks(np.arange(len(names)),names)
    plt.xticks(rotation=50)
    plt.ylim(0,1)
    plt.axhline(y=.5,color='k',linestyle='dashed')

    return

def symmetry_score_outliers(models):

    SList = [models[key][-1].flatten() for key in list(models.keys())]
    names = list(models.keys())

    plt.figure(figsize=(5,4))

    for i, model in enumerate(SList):
        
        outliers = count_outliers(SList[i].flatten())
        plt.scatter(i, outliers[outliers < .5].sum() / SList[i].size, color='darkblue')
        plt.scatter(i, outliers[outliers > .5].sum() / SList[i].size, color='purple')

    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.ylabel('# outliers')
    plt.xticks(np.arange(len(names)),names)
    plt.xticks(rotation = 50)
    plt.axhline(y = .0,color='k',linestyle='dashed')

    return

def plot_median_errorbars(parameters, scores, color = 'purple', alpha = 1.):

    for k in range(len(parameters)):

        scores_norm = 2 * scores[k] - 1

        median = np.median(scores_norm)
        q1_range = median - np.percentile(scores_norm, 25)
        q2_range = np.percentile(scores_norm, 75) - median

        plt.errorbar(parameters[k], median, yerr = [[q1_range], [q2_range]], 
                     fmt = 'o', ecolor = color, capsize = 5, marker = 'o', markersize = 5, color = color, alpha = alpha)

def plot_median_training(specs, models, colors = []):

    for idx, key in enumerate(list(specs.keys())):

        scores = 2 * models[key] - 1
        plt.plot(specs[key][0], np.median(scores, axis = (0,1)), color = colors[idx])
        plt.fill_between(specs[key][0], 
                np.percentile(scores, 25, axis = (0,1)),
                np.percentile(scores, 75, axis = (0,1)),
                color = colors[idx], alpha = .3)
        
def plot_median_initialization(data, layers, heads, c = 'navy', mode = 'full'):

    scores = np.zeros((layers, heads, len(data['_step'])))
    for l in range(layers):
            for h in range(heads):
                scores[l, h, :] = 2 * data[f'Layer {l}/Head {h} Symmetry WqWk'] - 1

    if mode == 'full':

        plt.plot(data['_step'], np.median(scores, axis = (0,1)), color = c)
        plt.fill_between(data['_step'], 
                np.percentile(scores, 25, axis = (0,1)),
                np.percentile(scores, 75, axis = (0,1)),
                color = c, alpha = .3)
    
    if mode == 'layers':

        for l in range(layers):
            plt.plot(data['_step'], np.median(scores, axis = 1)[l, :], color = c[l], label = f'layer {l}')
            plt.fill_between(data['_step'], 
                        np.percentile(scores, 25, axis = 1)[l, :],
                        np.percentile(scores, 75, axis = 1)[l, :],
                        color = c[l], alpha = .3)