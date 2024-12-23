import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def plot_median_scores(models, color = 'purple', mode = 'symmetry', alpha = 1.):
    """
    plots median scores with error bars for models based on symmetry or directionality

    parameters:
        models: dictionary containing model data, including scores and parameter counts
        color: string representing the color for the plot, default is 'purple'
        mode: string indicating the type of scores to plot ('symmetry' or 'directionality'), default is 'symmetry'
        alpha: float controlling the transparency of the plotted points, default is 1.0

    returns:
        none (displays a plot showing median scores with error bars for each model)
    """

    parameters = [models[key][-1][-1] for key in list(models.keys())]    
    if mode == 'symmetry':  
        scores = [2 * models[key][-3].flatten() - 1 for key in list(models.keys())]
    if mode == 'directionality':
        scores = [-1 * models[key][-2].flatten() for key in list(models.keys())]

    for k in range(len(parameters)):

        median = np.median(scores[k])
        q1_range = median - np.percentile(scores[k], 25)
        q2_range = np.percentile(scores[k], 75) - median

        plt.errorbar(parameters[k], median, yerr = [[q1_range], [q2_range]], 
                    fmt = 'o', ecolor = color, capsize = 5, marker = 'o', 
                    markersize = 5, color = color, alpha = alpha)
        
def plot_scores(models, color = 'purple', mode = 'symmetry', alpha = 1.):
    """
    plots median scores with error bars for models based on symmetry or directionality

    parameters:
        models: dictionary containing model data, including scores and parameter counts
        color: string representing the color for the plot, default is 'purple'
        mode: string indicating the type of scores to plot ('symmetry' or 'directionality'), default is 'symmetry'
        alpha: float controlling the transparency of the plotted points, default is 1.0

    returns:
        none (displays a plot showing median scores with error bars for each model)
    """

    checkpoints = len(models)
    score_median = np.zeros(checkpoints)
    score_q1 = np.zeros(checkpoints)
    score_q2 = np.zeros(checkpoints)

    for idx, key in enumerate(list(models.keys())):

        if mode == 'symmetry':  scores = 2 * models[key][-3].flatten() - 1
        elif mode == 'directionality': scores = -1 * models[key][-2].flatten()

        score_median[idx] = np.median(scores)
        score_q1[idx] = score_median[idx] - np.percentile(scores, 25)
        score_q2[idx] = np.percentile(scores, 75) - score_median[idx]

    plt.plot(score_median, color = color, linewidth = 2)
    plt.fill_between(range(checkpoints), score_median - score_q1, score_median + score_q2, color = color, alpha = .3)

def plot_scores_layers(models, color = 'purple', mode = 'symmetry', alpha = 1.):
    """
    plots median scores with error bars for models based on symmetry or directionality

    parameters:
        models: dictionary containing model data, including scores and parameter counts
        color: string representing the color for the plot, default is 'purple'
        mode: string indicating the type of scores to plot ('symmetry' or 'directionality'), default is 'symmetry'
        alpha: float controlling the transparency of the plotted points, default is 1.0

    returns:
        none (displays a plot showing median scores with error bars for each model)
    """

    layers = models['checkpoint-1000'][0].num_hidden_layers

    checkpoints = len(models)
    scores = np.zeros((layers, checkpoints))
    score_q1 = np.zeros((layers, checkpoints))
    score_q2 = np.zeros((layers, checkpoints))
    c = generate_color_list(layers, cmap = 'cividis')

    for idx, key in enumerate(list(models.keys())):

        if mode == 'symmetry':  scores[:, idx] = 2 * models[key][-3].flatten() - 1
        elif mode == 'directionality': scores[:, idx] = -1 * models[key][-2].flatten()\
        
    for l in range(layers):

        plt.plot(scores[l, :], color = c[l], linewidth = 2)

def generate_color_list(n, cmap):
    colormap = cm.get_cmap(cmap, n)
    print(colormap(0))
    color_list = [colormap(i) for i in range(n)]
    return color_list