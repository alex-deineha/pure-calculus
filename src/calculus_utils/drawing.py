import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter, get_common_distributions


def draw_hist(data):
    steps = [x for x in data if x != float("inf")]

    distributions = get_common_distributions()
    distributions.remove("expon")
    distributions.remove("cauchy")
    f = Fitter(steps, distributions=distributions)
    f.fit()
    summary = f.summary()
    distribution = f.get_best(method="sumsquare_error")

    print("==============")
    print(f"number of not normalized terms: {len(data) - len(steps)}")
    print(summary)
    print(distribution)
    print(f'Norm distribution: {f.fitted_param["norm"]}')
    print("==============")
    plt.xlabel("Number of reduction steps")
    plt.show()

    f_ln = Fitter([np.log(step) for step in steps], distributions=distributions)
    f_ln.fit()
    mu, sigma = f_ln.fitted_param["norm"]
    print(
        f"Log Norm distribution params: ({mu}, {sigma}); expected value = {np.e ** (mu + (sigma ** 2) / 2)}"
    )


def draw_steps_displot(data_to_draw: dict, kde=True, log_scale=(True, False), element: str = "bars",
                       x_label="steps", multiple: str = "layer", alpha=0.25, y_lines: tuple = None, bins=-1):
    """
    :param data_to_draw: a dict, where key is a name of a data seria,
                        and value is a list of the data seria (in this case steps)
    :param kde: bool, if it's True than draw density line
    :param log_scale: tutle of (bool, bool), where the 0-th set x-axis logarithmic scale,
                      and 1-th set y-axis logarithmic scale
    :param element: str of 'step' draw histograms as step-line
                    'poly' draw histograms as poly-angle figure (triangular form)
                    'bars' -- default histogram plot
    :param x_label: str of description x-label
    :param multiple: str of 'stack' -- plot histograms on top of each other
                     'dodge' -- plot histograms near each other
                     'fill' -- plot
                     'layer' -- default histogram plot
    :param alpha: float value of hist plot
    :param y_lines: tutle of expected values for each data seria
    :param bins: int, count of histograms, if -1 use default count of bins
    """

    if bins < 0:
        fig = sns.displot(data=data_to_draw, kde=kde, log_scale=log_scale,
                          element=element, multiple=multiple, alpha=alpha)
    else:
        fig = sns.displot(data=data_to_draw, kde=kde, log_scale=log_scale,
                          element=element, multiple=multiple, alpha=alpha, bins=bins)

    if y_lines:
        colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for inx, y_line in enumerate(y_lines):
            plt.axvline(x=y_line, color=colors_cycle[inx % len(colors_cycle)])

    fig.set_xlabels(label=x_label)
    plt.show()


def draw_cumulative_avg(
    steps, title="Cum avg for approachco", x_label="Trials", y_label="Average steps"
):
    cum_rewards = np.cumsum(steps)

    for i in range(len(cum_rewards)):
        cum_rewards[i] /= i + 1
    plt.plot(range(1, len(cum_rewards) + 1), cum_rewards)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def draw_cumulative_avg_comparison(
    comp_data: dict,
    title="Cum avg for approaches",
    x_label="Trials",
    y_label="Average steps",
):
    draw_mult_lines(comp_data, title=title, x_label=x_label, y_label=y_label)


def draw_mult_lines(
    comp_data: dict, title="title", x_label="x_label", y_label="y_label"
):
    for key in comp_data.keys():
        cum_reward = np.cumsum(comp_data[key][0])
        cum_reward = [cum_reward[i] / (i + 1) for i in range(len(cum_reward))]
        plt.plot(
            range(1, len(cum_reward) + 1), cum_reward, comp_data[key][1], label=key
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


def draw_2d_distribution(ax, x, y, xlabel, ylabel):
    ax.scatter(x, y, color="blue")
    ax.set(
        title="Distribution of " + xlabel.lower() + "-" + ylabel.lower(),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.legend(prop={"size": 10})


def draw_2d_strategy_distribution(ax, x, y, xlabel, ylabel):
    x = list(map(lambda v: -1 if v == float("inf") else v, x))
    y = list(map(lambda v: -1 if v == float("inf") else v, y))
    greater = list(
        zip(
            *list(
                filter(lambda z: z[0] >= z[1] and z[0] != -1 and z[1] != -1, zip(x, y))
            )
        )
    )
    less = list(
        zip(
            *list(
                filter(lambda z: z[0] < z[1] and z[0] != -1 and z[1] != -1, zip(x, y))
            )
        )
    )
    inf_x = list(zip(*list(filter(lambda z: z[0] == -1 and z[1] != -1, zip(x, y)))))
    inf_y = list(zip(*list(filter(lambda z: z[0] != -1 and z[1] == -1, zip(x, y)))))
    inf_xy = list(zip(*list(filter(lambda z: z[0] == -1 and z[1] == -1, zip(x, y)))))
    ax.scatter(
        greater[0], greater[1], color="blue", label="{} <= {}".format(ylabel, xlabel)
    )
    ax.scatter(less[0], less[1], color="red", label="{} < {}".format(xlabel, ylabel))
    if inf_x:
        ax.scatter(
            inf_x[0],
            inf_x[1],
            color="lime",
            label="{} doesn't normalize".format(xlabel),
        )
    if inf_y:
        ax.scatter(
            inf_y[0],
            inf_y[1],
            color="yellow",
            label="{} dooesn't normalize".format(ylabel),
        )
    if inf_xy:
        ax.scatter(
            inf_xy[0],
            inf_xy[1],
            color="orange",
            label="Both strategies don't normalize",
        )
    ax.set(
        title="Distribution of " + xlabel.lower() + "-" + ylabel.lower(),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.legend(prop={"size": 10})


def draw_plot(x, y, z, q, labels, colors):
    x = list(map(lambda v: -1 if v == float("inf") else v, x))
    y = list(map(lambda v: -1 if v == float("inf") else v, y))
    z = list(map(lambda v: -1 if v == float("inf") else v, z))
    q = list(map(lambda v: -1 if v == float("inf") else v, q))

    data = pd.DataFrame(zip(x, y, z, q), columns=labels)
    ax0 = data.plot(figsize=(20, 10), kind="bar", color=colors)
    ax0.set(
        title="Distribution of number of reduction steps for each term",
        xlabel="Term index",
        ylabel="Number of reduction steps",
    )
    ax0.legend(prop={"size": 10})
    plt.show()
    return ax0
