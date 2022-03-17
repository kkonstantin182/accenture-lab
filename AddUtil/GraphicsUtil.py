import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime

SIZE = 20
LINEWIDTH = 3
FIGSIZE = (10, 8)

class Visualizer():

    @staticmethod
    def plot_confsuion_matrix(y_test, y_pred, fig_size=FIGSIZE, font_size=SIZE ):

        conf_matrix = confusion_matrix(y_test, y_pred) #tn, fp, fn, tp

        fig, ax = plt.subplots(figsize=fig_size)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax = sns.heatmap(
                 conf_matrix/np.sum(conf_matrix), 
                 annot=True, fmt='.2%', 
                 cmap='Blues', 
                 annot_kws={"size": font_size})

        ax.set_title('Confusion Matrix\n\n', fontsize=font_size)
        ax.set_xlabel('\nPredicted Values', fontsize=font_size)
        ax.set_ylabel('Actual Values', fontsize=font_size)

        ax.xaxis.set_ticklabels(['False','True'], fontsize = font_size)
        ax.yaxis.set_ticklabels(['False','True'], fontsize = font_size)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size)

        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        fig.savefig(f'conf_mat_{date}.jpg', format='jpg')
