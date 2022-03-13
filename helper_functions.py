import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import Tuple, Dict, List, Callable


def calc_ratio(group: pd.DataFrame) -> pd.DataFrame:
    "Takes a grouped pandas DataFrame and returns DataFrame"

    group["ratio"] = (group["size"] / group["size"].sum() * 100).round(2)

    return group


def find_outliers(data_frame: pd.DataFrame, factor: float) -> Dict[str, str]:
    """Finds outliers in DataFrame and returns a dictionary with column names
    where outliers were found"""

    outliers_dict = {}
    for column in data_frame.columns:
        if data_frame[column].dtype not in [float, int]:
            continue
        if np.any(
            data_frame[column]
            > (data_frame[column].quantile(0.75) - data_frame[column].quantile(0.25))
            * factor
            + data_frame[column].quantile(0.75)
        ) or np.any(
            data_frame[column]
            < data_frame[column].quantile(0.25)
            - (data_frame[column].quantile(0.75) - data_frame[column].quantile(0.25))
            * factor
        ):
            outliers_dict[column] = True

    return outliers_dict


def quatile_sorting(series: pd.Series, measurement: str) -> Tuple[pd.Series, ...]:
    """Take series and returns a tuple of series with quantiles in a firts series
    and labels in a second series"""
    
    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]
    series_a = pd.qcut(series, quantile_list)
    quantile_labels = [
        f"{series_a.cat.categories[i].left} - {series_a.cat.categories[i].right} {measurement}"
        for i in range(len(series_a.cat.categories))
    ]
    series_b = pd.qcut(series, quantile_list, quantile_labels)

    return series_a, series_b


def two_p_diff_ci(
    series_a_r_count: int, series_b_r_count: int, total_count: int
) -> Tuple[float, ...]:
    """Takes series_a row count, series_b row count and total count of rows
    in a data set and returns difference in two proportions together with
    lower and upper bounds of 95 % confidence interval."""

    first_p = series_a_r_count / total_count
    second_p = series_b_r_count / total_count
    first_p_se = np.sqrt(first_p * (1 - first_p) / total_count)
    second_p_se = np.sqrt(second_p * (1 - second_p) / total_count)
    se_diff = np.sqrt(first_p_se ** 2 + second_p_se ** 2)
    diff_in_p = first_p - second_p
    lcb = diff_in_p - 1.96 * se_diff
    ucb = diff_in_p + 1.96 * se_diff

    return diff_in_p, lcb, ucb


def diff_in_two_means_unpooled_ci(
    series_a: pd.Series, series_b: pd.Series
) -> Tuple[float, ...]:
    """Takes series_a and series_b as an input and returns two series
    mean difference, lower and upper bounds of 95 % confidence interval"""

    s1_sem = series_a.std() / np.sqrt(series_a.count())
    s2_sem = series_b.std() / np.sqrt(series_b.count())
    sem_diff = np.sqrt(s1_sem ** 2 + s2_sem ** 2)
    mean_diff = series_a.mean() - series_b.mean()
    lcb = mean_diff - 2 * sem_diff
    ucb = mean_diff + 2 * sem_diff

    return mean_diff, lcb, ucb


def sampling_mean_diff_ci(
    series_a: pd.Series,
    series_b: pd.Series,
    number_of_samples: int,
    alpha: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """Takes two series, number of samples and alpha parameter and returns
    actual mean difference, sampled mean difference, lower and upper bounds
    of confidence interval controled by alpha parameter"""

    actual_mean_diff = series_a.mean() - series_b.mean()
    mean_diff = np.zeros((number_of_samples,))
    for i in range(number_of_samples):
        mean1 = np.random.choice(series_a, series_a.size).mean()
        mean2 = np.random.choice(series_b, series_b.size).mean()
        mean_diff[i] = mean1 - mean2
    quant = (1 - alpha) / 2
    lcb = np.quantile(mean_diff, quant)
    ucb = np.quantile(mean_diff, 1 - quant)

    return actual_mean_diff, mean_diff, lcb, ucb


def sampling_mean(
    series_a: pd.Series, series_b: pd.Series, sample_size: int, number_of_samples: int
) -> Tuple[np.ndarray]:
    """Samples number_of_samples from two series with with given sample_size
    and returns numpy arrays with means"""

    s1 = np.empty(sample_size)
    s2 = np.empty(sample_size)
    for i in range(number_of_samples):
        s1[i] = series_a.sample(sample_size, replace=True).mean()
        s2[i] = series_b.sample(sample_size, replace=True).mean()

    return s1, s2


def perm_test(series_a: pd.Series, series_b: pd.Series) -> float:
    """Makes a permutation test from provided input of two series
    and returns mean difference of permutation result"""

    s1 = series_a.reset_index(drop=True)
    s2 = series_b.reset_index(drop=True)
    data = pd.concat([s1, s2], ignore_index=True)
    total_n = s1.shape[0] + s2.shape[0]
    index2 = set(np.random.choice(range(total_n), s2.shape[0], replace=False))
    index1 = set(range(total_n)) - index2

    return data.iloc[list(index1)].mean() - data.iloc[list(index2)].mean()


def plot_mean_diff_conf_int(
    distribution: np.ndarray,
    lcb: float,
    actual_mean_diff: float,
    ucb: float,
    color: str = "k",
) -> plt.figure:
    """Plots a sampled distribution from sampling_mean_diff_ci function together
    with lower and upper bounds of sampled distribution condfidence intervals
    and actual mean of real distribution"""

    text_lst = [
        "Lower bound = ",
        "Upper bound = ",
        "Actual mean difference = ",
        "Sampled mean difference = ",
    ]
    dist_mean = distribution.mean()

    var_tup = (lcb, ucb, actual_mean_diff, dist_mean)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax = sns.kdeplot(distribution)

    ax.axvline(lcb, ls="--", color="b")
    ax.axvline(ucb, ls="--", color="b")
    ax.axvline(actual_mean_diff, color="IndianRed")
    ax.axvline(dist_mean, color="y")

    y = ax.get_ylim()[1] * 0.5

    for i in range(len(text_lst)):
        x_text = 4
        if i == 0: x_text = -12
        elif i == 2: x_text = -25
        elif i == 3: x_text = 18
        ax.annotate(
            f"{text_lst[i]}{var_tup[i]:.3f}",
            (var_tup[i], y),
            rotation="90",
            va="center",
            color=color,
            xytext=(x_text, 0),
            size=12,
            textcoords="offset points",
        )

    kde = ax.get_lines()[0].get_data()
    ax.fill_between(
        kde[0],
        kde[1],
        where=(kde[0] > lcb) & (kde[0] < ucb),
        interpolate=True,
        color="Lavender",
    )

    ax.set_title(
        "Sampled distribution of difference in means with 95 % CI", fontsize=13, y=1.01
    )

    plt.show()


def plot_coefficients(
    model_coef_list: List[Pipeline],
    leg_labels_list: List[str],
    x_labels: List[str] = None,
    title: str = None,
    y_scale: str = None,
) -> plt.figure:
    """Plots up to 7 different models coeficients on the same plot"""

    markers = ["o", "v", "s", "p", "^", "<", ">"]

    if len(model_coef_list) > 7:
        return f"Too many models to plot"

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(model_coef_list)):

        ax.plot(
            model_coef_list[i],
            marker=markers[i],
            c=np.random.rand(
                3,
            ),
            linestyle="None",
        )

    ax.legend(
        leg_labels_list, bbox_to_anchor=(1.35, 0.5), edgecolor="white", fontsize=12
    )

    ax.set_xticks(range(len(model_coef_list[i])))
    if x_labels != None:
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(range(len(model_coef_list[0])))
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_ylabel("Coefficient magnitude")
    ax.set_title(title, fontsize=13, y=1.03)
    if y_scale != None:
        ax.set_yscale(y_scale)
        mn = min(list(map(min, model_coef_list)))
        mx = max(list(map(max, model_coef_list)))
        ax.set_ylim(mn + mn * 0.3, mx + mx * 0.3)

    plt.show()


def plot_cm(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    display_labels: List[str] = None,
    cmap: str = None,
    colorbar: bool = None,
    title: str = None,
) -> plt.figure:
    """Plots a confusion matrix of a classification model"""

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("font", size=12)

    if colorbar == None:
        colorbar = False

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        display_labels=display_labels,
        colorbar=colorbar,
        cmap=cmap,
        ax=ax,
    )
    plt.title(
        title,
        fontsize=13,
        y=1.03,
    )
    plt.grid(False)

    plt.show()


def plot_multioutput_cm(
    model_lst: List[Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_lst: List[str] = None,
    cmap: str = None,
) -> plt.figure:
    "Plots multioutput classification results"

    fig, axes = plt.subplots(1, 2, figsize=(6 * len(model_lst), 6), sharey=True)

    plt.rc("font", size=12)

    for i, ax in enumerate(axes.flatten()):

        plot_confusion_matrix(
            model_lst[i],
            X_test,
            y_test.iloc[:, i],
            colorbar=False,
            cmap=cmap,
            ax=ax,
        )
        ax.set_title(
            title_lst[i],
            fontsize=13,
            y=1.03,
        )
        ax.grid(False)

    plt.show()


def plot_cm_without_model(
    Y: np.ndarray,
    Y_predicted: np.ndarray,
    display_labels: List[str] = None,
    cmap: str = None,
    colorbar: bool = None,
    title: str = None,
) -> plt.figure:
    """Plots a confusion matrix without a model"""

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("font", size=12)

    if colorbar == None:
        colorbar = False

    cm = confusion_matrix(Y, Y_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        colorbar=colorbar,
        cmap=cmap,
        ax=ax,
    )
    plt.title(
        title,
        fontsize=13,
        y=1.03,
    )
    plt.grid(False)

    plt.show()


def plot_residuals(
    values_list: List[np.ndarray],
    errors_lst: List[np.ndarray],
    x_labels_list: List[str],
    title_list: List[str],
) -> None:
    """From input lists, plots residuals plots against train and test data"""

    fig, axes = plt.subplots(1, len(values_list), figsize=(16, 7), sharey=True)

    for i, ax in enumerate(axes.flatten()):

        sns.scatterplot(x=values_list[i], y=errors_lst[i], alpha=0.5, ax=ax)
        lin = ax.axhline(0, color="r")
        ax.set_xlabel(x_labels_list[i], fontsize=12, labelpad=10)
        ax.set_ylabel("Residuals", fontsize=12, labelpad=10)
        ax.set_title(title_list[i], fontsize=13, y=1.03)

    ax.legend(
        [lin],
        ["Perfect guess by a model"],
        bbox_to_anchor=(0.13, -0.13),
        edgecolor="white",
        fontsize=12,
    )


def plot_reg(
    prediction: np.ndarray,
    y_test_data: np.ndarray,
    title: str = None,
    ylabel: str = None,
) -> plt.figure:
    """
    Plots a lineplot for true y_test values and a scatter plot for predictions
    """

    fig, ax = plt.subplots(figsize=(20, 8))

    x_points = range(len(y_test_data))
    sns.lineplot(
        x=x_points, y=y_test_data, color="b", lw=2, label="Test data values", ax=ax
    )
    sns.scatterplot(
        x=x_points, y=prediction, alpha=0.7, color="red", label="Predicted data", ax=ax
    )
    ax.set_xlabel("Data points", fontsize=13, labelpad=10)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, y=1.03)
    ax.legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.13),
        edgecolor="white",
        fontsize=13,
    )

    plt.show()


def catplot_with_pct(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str
) -> plt.figure:
    "Plots categorical data with percentage"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        data=df,
        kind="bar",
        ci=None,
        height=6,
        aspect=2,
        palette="husl",
    )

    g._legend.set_title(f"{hue.split('_')[0].capitalize()} category")

    ax = g.axes.flat[0]
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            size=11,
            textcoords="offset points",
        )

    g.axes.flat[0].get_yaxis().set_visible(False)
    g.axes.flat[0].tick_params(axis="x", which="major", labelsize=12)
    g.fig.suptitle(title, fontsize=13, y=1.03)
    plt.setp(g._legend.get_title(), fontsize=12)

    plt.show()


def bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: str,
    title: str,
    leg_title: str = None,
) -> plt.figure:
    "Plots bar plots"

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        ci=None,
        palette=palette,
        dodge=False,
        ax=ax,
    )
    if not leg_title:
        leg_title = hue.split("_")[0].capitalize()
    ax.legend(
        title=leg_title,
        fontsize=12,
        loc="center",
        ncol=1,
        bbox_to_anchor=(1.2, 0.5),
        facecolor="white",
        edgecolor="white",
    )

    ax.set_xlabel("")
    ax.set_ylabel(" ".join(y.split("_")).capitalize(), fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)

    plt.title(
        title,
        fontsize=13,
        y=1.02,
    )

    plt.show()


def grouped_bar_plots(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str = None
) -> plt.figure:
    "Plots grouped bar plots"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        data=df,
        kind="bar",
        ci=None,
        height=8,
        aspect=2.5,
        palette="Set2",
    )

    g._legend.set_title("")

    for t in g._legend.texts:
        t.set_fontsize(14)

    ax = g.axes.flat[0]
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(4, 5),
            size=14,
            textcoords="offset points",
        )

    g.axes.flat[0].set_xticklabels(
        g.axes.flat[0].get_xticklabels(), rotation=30, ha="right", fontsize=14
    )
    g.axes.flat[0].get_yaxis().set_visible(False)
    g.fig.suptitle(
        title,
        fontsize=18,
        y=1.03,
    )

    plt.show()


def stacked_bars(
    df: pd.DataFrame,
    y1: str,
    y2: str,
    teams_lst: List[str],
    legend_labels_lst: List[str],
    color1: str,
    color2: str,
    title: str,
) -> plt.figure:
    "Plots stacked bar plots"

    fig, ax = plt.subplots(figsize=(12, 8))

    bar1 = sns.barplot(
        x=[i for i in range(df.shape[0])],
        y=y1,
        data=df,
        color=color1,
    )

    bar2 = sns.barplot(
        x=[i for i in range(df.shape[0])],
        y=y2,
        data=df,
        color=color2,
    )

    ax.set_xticklabels(teams_lst, rotation=45, fontsize=12, ha="right")
    ax.set_yticks(range(max(df[y1]) + 1))
    ax.set_ylabel("Goals", fontsize=12, labelpad=30, rotation=0)

    top_bar = mpatches.Patch(color=color1, label=legend_labels_lst[0])
    bottom_bar = mpatches.Patch(color=color2, label=legend_labels_lst[1])

    ax.legend(
        handles=[top_bar, bottom_bar],
        fontsize=12,
        loc="center",
        ncol=1,
        bbox_to_anchor=(1.2, 0.5),
        facecolor="white",
        edgecolor="white",
    )

    ax.set_title(
        title,
        fontsize=13,
        y=1.03,
    )

    plt.show()


def heat_map(df: pd.DataFrame, title: str = None, cmap: str = None) -> plt.figure:
    "Plots a heatmap"

    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(14, 10))

    ax = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap=cmap)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlation Score", labelpad=15)
    ax.figure.axes[-1].yaxis.label.set_size(14)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)

    plt.show()


def create_reversed_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**(-1)
    and return it as series"""

    rev_series = np.power(series, -1)
    return rev_series


def create_log_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of log(series)
    and return it as series"""

    log_series = np.log((series + 1e-4))
    return log_series


def create_squared_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**2
    and return it as series"""

    sq_series = np.power(series, 2)
    return sq_series


def create_cubic_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**3
    and return it as series"""

    cub_series = np.power(series, 3)
    return cub_series


def create_product_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A*B
    and return it as series"""

    product_series = np.multiply(series_a, series_b)
    return product_series


def create_division_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A/B
    and return it as series"""

    div_series = np.divide(series_a, (series_b + 1e-4))
    return div_series


def create_addition_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A+B
    and return it as series"""

    add_series = np.add(series_a, series_b)
    return add_series


def create_subtraction_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A-B
    and return it as series"""

    sub_series = np.subtract(series_a, series_b)
    return sub_series


def single_column_transformation(
    data_frame: pd.DataFrame, target_series: pd.Series, functions_list: List[Callable]
) -> pd.DataFrame:
    """From input data_frame creates a new features via provided transforming
    function list. Returns a new dataframe with transformed features."""

    df = data_frame.copy()
    new_features_lst = []
    new_features_col_names = []

    for i in range(len(functions_list)):
        for j in range(len(df.columns)):
            if df.iloc[:, j].dtype != float:
                df.loc[:, df.columns[j]] = df.loc[:, df.columns[j]].astype(float)
            new_feature = functions_list[i](df.iloc[:, j])

            original_correlation = np.abs(df.iloc[:, j].corr(target_series))
            new_feature_correlation = np.abs(new_feature.corr(target_series))

            if new_feature_correlation > original_correlation:
                new_feature_name = (
                    f"{df.columns[j]}_{str(functions_list[i]).split('_')[-2]}"
                )
                new_features_lst.append(new_feature)
                new_features_col_names.append(new_feature_name)

    return pd.concat(new_features_lst, axis=1, keys=new_features_col_names)


def pairwise_transformations(
    data_frame: pd.DataFrame, target_series: pd.Series, functions_list: List[Callable]
) -> pd.DataFrame:
    """From input data_frame creates a new features via provided transforming
    function list. Returns a new dataframe with transformed features."""

    df = data_frame.copy()
    new_features_lst = []
    new_features_col_names = []

    for i in range(len(functions_list)):
        for j in range(len(df.columns)):
            for k in range(len(df.columns)):
                if df.columns[j] != df.columns[k]:
                    if df.iloc[:, j].dtype != float or df.iloc[:, k].dtype != float:
                        df.loc[:, df.columns[j]] = df.loc[:, df.columns[j]].astype(
                            float
                        )
                        df.loc[:, df.columns[k]] = df.loc[:, df.columns[k]].astype(
                            float
                        )

                    new_feature = functions_list[i](df.iloc[:, j], df.iloc[:, k])

                    original_correlation1 = np.abs(df.iloc[:, j].corr(target_series))
                    original_correlation2 = np.abs(df.iloc[:, k].corr(target_series))
                    bigest_original_corr = max(
                        original_correlation1, original_correlation2
                    )
                    new_feature_correlation = np.abs(new_feature.corr(target_series))

                    if new_feature_correlation > bigest_original_corr:
                        new_feature_name = f"{df.columns[j]}_{str(functions_list[i]).split('_')[-2]}_{df.columns[k]}"
                        new_features_lst.append(new_feature)
                        new_features_col_names.append(new_feature_name)

    return pd.concat(new_features_lst, axis=1, keys=new_features_col_names)


def get_duplicate_columns(data_frame: pd.Series) -> List[str]:
    """From input dataframe finds columns which are the same
    content wise and returns those columns names as a list."""

    duplicate_column_names = set()

    for i in range(data_frame.shape[1]):
        col = data_frame.iloc[:, i]
        for j in range(i + 1, data_frame.shape[1]):
            other_col = data_frame.iloc[:, j]

            if col.equals(other_col):
                duplicate_column_names.add(data_frame.columns.values[j])

    return list(duplicate_column_names)


def adj_threshold_to_labels(model_probs: np.ndarray, threshold: float) -> np.ndarray:
    """From input of positive class probabilites applies
    threshold to positive probabilities to create labels"""

    return (model_probs >= threshold).astype("int")
