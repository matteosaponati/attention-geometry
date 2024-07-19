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

def plot_median_errorbars(parameters, scores, color = 'purple'):

    for k in range(len(parameters)):

        median = np.median(scores[k])
        q1_range = median - np.percentile(scores[k], 25)
        q2_range = np.percentile(scores[k], 75) - median

        plt.errorbar(parameters[k], median, yerr = [[q1_range], [q2_range]], 
                     fmt = 'o', ecolor = color, capsize = 5, marker = 'o', markersize = 5, color = color)