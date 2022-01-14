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
    #labels = {'action': ('F.A. Action Only', 'blue'),
    #          'scalar': ('F.A. Scalar Only', 'red'),
    #          'human_choice': ('F.A. Human Choice', 'green'),
    #          'full': ('F.R./All Human Choice', 'magenta'),
    #          'traj': ('F.R./Cur Human Choice', 'gold'),
    #          'baseline': ('Baseline', 'cyan')}
    #exps = ["action", "scalar", "human_choice", "full", "traj", "baseline"]
    color_list = ['C2', 'C0', 'C4', 'C6', 'C1', 'C3', 'C5']
    labels = {'0': ('1.0 E, 0.0 A', 'C2'),
              '1': ('0.9 E, 0.1 A', 'C0'),
              '2': ('0.7 E, 0.3 A', 'C4'),
              '3': ('0.5 E, 0.5 A', 'C6'),
              '4': ('0.3 E, 0.7 A', 'C1'),
              '5': ('0.1 E, 0.9 A', 'C3'),
              '6': ('0.0 E, 1.0 A', 'C5')}
    exps = ['0','1','2','3','4','5','6']
    color_list = ['C2', 'C0', 'C4', 'C6', 'C1', 'C3', 'C5']
    labels = {'0': ('0', '#6bd0f3'),
              '1': ('1', '#077ccc'),
              '2': ('2', '#075791'),
              '3': ('3', '#AA0000'),
              '4': ('4', 'black'),
              '5': ('', 'white'),
              '6': ('', 'white')}
    exps = ['0','1','2','3','4','5','6']

    export_legend(labels, exps, "reward_legend.pdf")
