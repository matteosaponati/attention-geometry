import numpy as np
import matplotlib.pyplot as plt

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
            
# def plot_median_training(specs, models, colors = []):

#     for idx, key in enumerate(list(specs.keys())):

#         scores = 2 * models[key] - 1
#         plt.plot(specs[key][0], np.median(scores, axis = (0,1)), color = colors[idx])
#         plt.fill_between(specs[key][0], 
#                 np.percentile(scores, 25, axis = (0,1)),
#                 np.percentile(scores, 75, axis = (0,1)),
#                 color = colors[idx], alpha = .3)
        
# def plot_median_initialization(data, layers, heads, c = 'navy', mode = 'full'):

#     scores = np.zeros((layers, heads, len(data['_step'])))
#     for l in range(layers):
#             for h in range(heads):
#                 scores[l, h, :] = 2 * data[f'Layer {l}/Head {h} Symmetry WqWk'] - 1

#     if mode == 'full':

#         plt.plot(data['_step'], np.median(scores, axis = (0,1)), color = c)
#         plt.fill_between(data['_step'], 
#                 np.percentile(scores, 25, axis = (0,1)),
#                 np.percentile(scores, 75, axis = (0,1)),
#                 color = c, alpha = .3)
    
#     if mode == 'layers':

#         for l in range(layers):
#             plt.plot(data['_step'], np.median(scores, axis = 1)[l, :], color = c[l], label = f'layer {l}')
#             plt.fill_between(data['_step'], 
#                         np.percentile(scores, 25, axis = 1)[l, :],
#                         np.percentile(scores, 75, axis = 1)[l, :],
#                         color = c[l], alpha = .3)