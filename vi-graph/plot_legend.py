import matplotlib.pyplot as plt

def export_legend(labels, exps, savepath):
    """
    Export a legend in a borderless, horizontal format
    """
    plt.rcParams.update({'font.size': 20})
    ## Create a fake axes to use
    x = [0, 1]
    y = [0, 1]
    for exp in exps:
        plt.plot(x,y,linewidth=3, label=labels[exp][0], color=labels[exp][1])
    ax = plt.gca()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=len(exps),)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(savepath, dpi="figure", bbox_inches=bbox)

if __name__ == "__main__":
    labels = {'action': ('F.A. Action Only', 'blue'),
              'scalar': ('F.A. Scalar Only', 'red'),
              'human_choice': ('F.A. Human Choice', 'green'),
              'full': ('F.R./All Human Choice', 'magenta'),
              'traj': ('F.R./Cur Human Choice', 'gold'),
              'baseline': ('Baseline', 'cyan')}
    exps = ["action", "scalar", "human_choice", "full", "traj", "baseline"]

    export_legend(labels, exps, "legend.pdf")
