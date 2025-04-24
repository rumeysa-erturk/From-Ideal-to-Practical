import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


### Filter by variant caller
def FetchScoresVariantCallerBinaryFiltering(df, values, column):
    """
    Fetches the precision, recall, and f1 scores for the two categories in the binary filtering
    """

    cat_1 = df[df[column] == values[0]]
    cat_2 = df[df[column] == values[1]]

    #######
    cat1_mutect = cat_1[cat_1["pipeline"].str.startswith("mutect")]
    cat1_strelka = cat_1[cat_1["pipeline"].str.startswith("strelka")]
    cat1_ss = cat_1[cat_1["pipeline"].str.startswith("ss")]

    cat2_mutect = cat_2[cat_2["pipeline"].str.startswith("mutect")]
    cat2_strelka = cat_2[cat_2["pipeline"].str.startswith("strelka")]
    cat2_ss = cat_2[cat_2["pipeline"].str.startswith("ss")]

    #######

    cat1_mutect_precisions = cat1_mutect["precision"]
    cat1_strelka_precisions = cat1_strelka["precision"]
    cat1_ss_precisions = cat1_ss["precision"]

    cat2_mutect_precisions = cat2_mutect["precision"]
    cat2_strelka_precisions = cat2_strelka["precision"]
    cat2_ss_precisions = cat2_ss["precision"]

    cat1_mutect_recalls = cat1_mutect["recall"]
    cat1_strelka_recalls = cat1_strelka["recall"]
    cat1_ss_recalls = cat1_ss["recall"]

    cat2_mutect_recalls = cat2_mutect["recall"]
    cat2_strelka_recalls = cat2_strelka["recall"]
    cat2_ss_recalls = cat2_ss["recall"]

    cat1_mutect_f1s = cat1_mutect["f1"]
    cat1_strelka_f1s = cat1_strelka["f1"]
    cat1_ss_f1s = cat1_ss["f1"]

    cat2_mutect_f1s = cat2_mutect["f1"]
    cat2_strelka_f1s = cat2_strelka["f1"]
    cat2_ss_f1s = cat2_ss["f1"]

    scores_cat1 = {
        "mutect-precision": cat1_mutect_precisions,
        "strelka-precision": cat1_strelka_precisions,
        "ss-precision": cat1_ss_precisions,
        "mutect-recall": cat1_mutect_recalls,
        "strelka-recall": cat1_strelka_recalls,
        "ss-recall": cat1_ss_recalls,
        "mutect-f1": cat1_mutect_f1s,
        "strelka-f1": cat1_strelka_f1s,
        "ss-f1": cat1_ss_f1s,
    }

    scores_cat2 = {
        "mutect-precision": cat2_mutect_precisions,
        "strelka-precision": cat2_strelka_precisions,
        "ss-precision": cat2_ss_precisions,
        "mutect-recall": cat2_mutect_recalls,
        "strelka-recall": cat2_strelka_recalls,
        "ss-recall": cat2_ss_recalls,
        "mutect-f1": cat2_mutect_f1s,
        "strelka-f1": cat2_strelka_f1s,
        "ss-f1": cat2_ss_f1s,
    }

    return scores_cat1, scores_cat2


## Filter by Mapper
def FetchScoresMapperBinaryFiltering(df, values, column):
    """
    Fetches the precision, recall, and f1 scores for the two categories in the binary filtering
    """

    cat_1 = df[df[column] == values[0]]
    cat_2 = df[df[column] == values[1]]

    #######
    cat1_bwa = cat_1[cat_1["pipeline"].str.contains("bwa")]
    cat1_bowtie = cat_1[cat_1["pipeline"].str.contains("bowtie")]

    cat2_bwa = cat_2[cat_2["pipeline"].str.contains("bwa")]
    cat2_bowtie = cat_2[cat_2["pipeline"].str.contains("bowtie")]

    #######

    cat1_bwa_precisions = cat1_bwa["precision"]
    cat1_bowtie_precisions = cat1_bowtie["precision"]
    cat2_bwa_precisions = cat2_bwa["precision"]
    cat2_bowtie_precisions = cat2_bowtie["precision"]

    cat1_bwa_recalls = cat1_bwa["recall"]
    cat1_bowtie_recalls = cat1_bowtie["recall"]
    cat2_bwa_recalls = cat2_bwa["recall"]
    cat2_bowtie_recalls = cat2_bowtie["recall"]

    cat1_bwa_f1s = cat1_bwa["f1"]
    cat1_bowtie_f1s = cat1_bowtie["f1"]
    cat2_bwa_f1s = cat2_bwa["f1"]
    cat2_bowtie_f1s = cat2_bowtie["f1"]

    scores_cat1 = {
        "bwa-precision": cat1_bwa_precisions,
        "bowtie-precision": cat1_bowtie_precisions,
        "bwa-recall": cat1_bwa_recalls,
        "bowtie-recall": cat1_bowtie_recalls,
        "bwa-f1": cat1_bwa_f1s,
        "bowtie-f1": cat1_bowtie_f1s,
    }

    scores_cat2 = {
        "bwa-precision": cat2_bwa_precisions,
        "bowtie-precision": cat2_bowtie_precisions,
        "bwa-recall": cat2_bwa_recalls,
        "bowtie-recall": cat2_bowtie_recalls,
        "bwa-f1": cat2_bwa_f1s,
        "bowtie-f1": cat2_bowtie_f1s,
    }

    return scores_cat1, scores_cat2


def FetchScoresBinaryFilteringAverage(df, values, column):
    """
    Fetches the average precision, recall, and f1 scores for the two categories in the binary filtering
    """

    cat_1 = df[df[column] == values[0]]
    cat_2 = df[df[column] == values[1]]

    #######
    cat1_mutect = cat_1[cat_1["pipeline"].str.startswith("mutect")]
    cat1_strelka = cat_1[cat_1["pipeline"].str.startswith("strelka")]
    cat1_ss = cat_1[cat_1["pipeline"].str.startswith("ss")]

    cat2_mutect = cat_2[cat_2["pipeline"].str.startswith("mutect")]
    cat2_strelka = cat_2[cat_2["pipeline"].str.startswith("strelka")]
    cat2_ss = cat_2[cat_2["pipeline"].str.startswith("ss")]
    #######

    avg_prec_cat1_mutect = cat1_mutect["precision"].mean()
    avg_prec_cat1_strelka = cat1_strelka["precision"].mean()
    avg_prec_cat1_ss = cat1_ss["precision"].mean()

    avg_recall_cat1_mutect = cat1_mutect["recall"].mean()
    avg_recall_cat1_strelka = cat1_strelka["recall"].mean()
    avg_recall_cat1_ss = cat1_ss["recall"].mean()

    avg_f1_cat1_mutect = cat1_mutect["f1"].mean()
    avg_f1_cat1_strelka = cat1_strelka["f1"].mean()
    avg_f1_cat1_ss = cat1_ss["f1"].mean()

    ##

    avg_prec_cat2_mutect = cat2_mutect["precision"].mean()
    avg_prec_cat2_strelka = cat2_strelka["precision"].mean()
    avg_prec_cat2_ss = cat2_ss["precision"].mean()

    avg_recall_cat2_mutect = cat2_mutect["recall"].mean()
    avg_recall_cat2_strelka = cat2_strelka["recall"].mean()
    avg_recall_cat2_ss = cat2_ss["recall"].mean()

    avg_f1_cat2_mutect = cat2_mutect["f1"].mean()
    avg_f1_cat2_strelka = cat2_strelka["f1"].mean()
    avg_f1_cat2_ss = cat2_ss["f1"].mean()

    #####

    mutects_cat1 = [avg_prec_cat1_mutect, avg_recall_cat1_mutect, avg_f1_cat1_mutect]
    strelkas_cat1 = [
        avg_prec_cat1_strelka,
        avg_recall_cat1_strelka,
        avg_f1_cat1_strelka,
    ]
    sss_cat1 = [avg_prec_cat1_ss, avg_recall_cat1_ss, avg_f1_cat1_ss]
    scores_cat1 = {"mutect": mutects_cat1, "strelka": strelkas_cat1, "ss": sss_cat1}

    mutects_cat2 = [avg_prec_cat2_mutect, avg_recall_cat2_mutect, avg_f1_cat2_mutect]
    strelkas_cat2 = [
        avg_prec_cat2_strelka,
        avg_recall_cat2_strelka,
        avg_f1_cat2_strelka,
    ]
    sss_cat2 = [avg_prec_cat2_ss, avg_recall_cat2_ss, avg_f1_cat2_ss]
    scores_cat2 = {"mutect": mutects_cat2, "strelka": strelkas_cat2, "ss": sss_cat2}

    return scores_cat1, scores_cat2


def PlotBinaryCategoryScoresBarChart(scores_cat1, scores_cat2, labels, title, figname):

    score_names = ["Precision", "Recall", "F1"]
    # X positions for the groups
    index = np.arange(3)  # Array with elements [0, 1, 2, 3, 4]

    # Width of a bar
    bar_width = 0.1

    plt.figure(figsize=(12, 8))

    # Creating the bars
    plt.bar(
        index,
        scores_cat1["mutect"],
        width=bar_width,
        label=labels[0],
        color="lightcoral",
    )
    plt.bar(
        index + bar_width,
        scores_cat2["mutect"],
        width=bar_width,
        label=labels[1],
        color="indianred",
    )

    plt.bar(
        index + 2 * bar_width,
        scores_cat1["strelka"],
        width=bar_width,
        label=labels[2],
        color="turquoise",
    )
    plt.bar(
        index + 3 * bar_width,
        scores_cat2["strelka"],
        width=bar_width,
        label=labels[3],
        color="lightseagreen",
    )

    plt.bar(
        index + 4 * bar_width,
        scores_cat1["ss"],
        width=bar_width,
        label=labels[4],
        color="mediumslateblue",
    )
    plt.bar(
        index + 5 * bar_width,
        scores_cat2["ss"],
        width=bar_width,
        label=labels[5],
        color="rebeccapurple",
    )

    # Adding labels, title, and custom x-axis tick labels
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title(title, fontsize=20)
    plt.xticks(
        index + 2 * bar_width, score_names
    )  # Positioning the category label in between the bars
    plt.legend(fontsize=10)

    # Show the plot
    plt.savefig(os.path.join(DIRECTORY, "Merged_Analysis_Figures", figname + ".png"))
    plt.show()


def PlotBinaryCategoryVariantCallerScoresBoxPlot(
    scores_cat1, scores_cat2, labels_, title, figname
):

    score_names = ["Precision", "Recall", "F1"]

    # Creating the boxplot

    fig, ax = plt.subplots(figsize=(20, 12))

    # Precisions
    mutect_cat1 = ax.boxplot(
        scores_cat1["mutect-precision"],
        positions=[1],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    mutect_cat2 = ax.boxplot(
        scores_cat2["mutect-precision"],
        positions=[1.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="indianred"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat1 = ax.boxplot(
        scores_cat1["strelka-precision"],
        positions=[1.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="turquoise"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat2 = ax.boxplot(
        scores_cat2["strelka-precision"],
        positions=[1.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightseagreen"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat1 = ax.boxplot(
        scores_cat1["ss-precision"],
        positions=[1.8],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumslateblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat2 = ax.boxplot(
        scores_cat2["ss-precision"],
        positions=[2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="rebeccapurple"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Recalls
    mutect_cat1 = ax.boxplot(
        scores_cat1["mutect-recall"],
        positions=[3],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    mutect_cat2 = ax.boxplot(
        scores_cat2["mutect-recall"],
        positions=[3.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="indianred"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat1 = ax.boxplot(
        scores_cat1["strelka-recall"],
        positions=[3.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="turquoise"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat2 = ax.boxplot(
        scores_cat2["strelka-recall"],
        positions=[3.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightseagreen"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat1 = ax.boxplot(
        scores_cat1["ss-recall"],
        positions=[3.8],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumslateblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat2 = ax.boxplot(
        scores_cat2["ss-recall"],
        positions=[4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="rebeccapurple"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # F1s
    mutect_cat1 = ax.boxplot(
        scores_cat1["mutect-f1"],
        positions=[5],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    mutect_cat2 = ax.boxplot(
        scores_cat2["mutect-f1"],
        positions=[5.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="indianred"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat1 = ax.boxplot(
        scores_cat1["strelka-f1"],
        positions=[5.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="turquoise"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    strelka_cat2 = ax.boxplot(
        scores_cat2["strelka-f1"],
        positions=[5.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="lightseagreen"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat1 = ax.boxplot(
        scores_cat1["ss-f1"],
        positions=[5.8],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumslateblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ss_cat2 = ax.boxplot(
        scores_cat2["ss-f1"],
        positions=[6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="rebeccapurple"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    ax.legend(
        [
            mutect_cat1["boxes"][0],
            mutect_cat2["boxes"][0],
            strelka_cat1["boxes"][0],
            strelka_cat2["boxes"][0],
            ss_cat1["boxes"][0],
            ss_cat2["boxes"][0],
        ],
        labels_,
        fontsize=25,
        bbox_to_anchor=(1, 1),
    )

    plt.xticks([1.5, 3.5, 5.5], score_names)

    # Adding labels, title, and custom x-axis tick labels
    plt.xlabel("Metrics")
    plt.ylabel("Scores")

    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Show the plot
    plt.savefig(
        os.path.join(DIRECTORY, "Merged_Analysis_Figures", figname + ".pdf"),
        bbox_inches="tight",
    )
    plt.show()


def PlotBinaryCategoryMapperScoresBoxPlot(
    scores_cat1, scores_cat2, labels_, title, figname
):

    score_names = ["Precision", "Recall", "F1"]

    # Creating the boxplot
    plt.rcParams.update({"font.size": 25})
    fig, ax = plt.subplots(figsize=(20, 12))

    # Precisions
    bwa_cat1 = ax.boxplot(
        scores_cat1["bwa-precision"],
        positions=[1],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="sandybrown"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bwa_cat2 = ax.boxplot(
        scores_cat2["bwa-precision"],
        positions=[1.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="chocolate"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat1 = ax.boxplot(
        scores_cat1["bowtie-precision"],
        positions=[1.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="cornflowerblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat2 = ax.boxplot(
        scores_cat2["bowtie-precision"],
        positions=[1.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Recalls
    bwa_cat1 = ax.boxplot(
        scores_cat1["bwa-recall"],
        positions=[3],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="sandybrown"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bwa_cat2 = ax.boxplot(
        scores_cat2["bwa-recall"],
        positions=[3.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="chocolate"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat1 = ax.boxplot(
        scores_cat1["bowtie-recall"],
        positions=[3.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="cornflowerblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat2 = ax.boxplot(
        scores_cat2["bowtie-recall"],
        positions=[3.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # F1s
    bwa_cat1 = ax.boxplot(
        scores_cat1["bwa-f1"],
        positions=[5],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="sandybrown"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bwa_cat2 = ax.boxplot(
        scores_cat2["bwa-f1"],
        positions=[5.2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="chocolate"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat1 = ax.boxplot(
        scores_cat1["bowtie-f1"],
        positions=[5.4],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="cornflowerblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bowtie_cat2 = ax.boxplot(
        scores_cat2["bowtie-f1"],
        positions=[5.6],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="mediumblue"),
        medianprops=dict(color="black", linewidth=1.5),
    )

    ax.legend(
        [
            bwa_cat1["boxes"][0],
            bwa_cat2["boxes"][0],
            bowtie_cat1["boxes"][0],
            bowtie_cat2["boxes"][0],
        ],
        labels_,
        bbox_to_anchor=(1, 1),
        fontsize="25",
    )

    plt.xticks([1.5, 3.5, 5.5], score_names)

    # Adding labels, title, and custom x-axis tick labels
    plt.xlabel("Metrics")
    plt.ylabel("Scores")

    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Show the plot
    plt.savefig(
        os.path.join(DIRECTORY, "Merged_Analysis_Figures", figname + ".png"),
        bbox_inches="tight",
    )
    plt.show()


def PlotScoresAsStripplot(melted_df):

    plt.figure(figsize=(16, 10))
    label_dict = {
        0: "Analysis",
        1: "Downloading the data",
        2: "Installation",
        3: "Mapping",
        4: "Variant Calling",
    }
    melted_df["Most difficult part"] = melted_df["Most difficult part"].map(label_dict)
    sns.stripplot(
        data=melted_df,
        x="metric",
        y="score",
        hue="Most difficult part",
        dodge=True,
        jitter=False,
    )
    plt.ylabel("Scores")
    plt.xlabel("Metrics")
    plt.title("Scores of the pipelines based on the most difficult part")
    plt.legend(fontsize=10)
    plt.savefig(
        os.path.join(
            DIRECTORY, "Merged_Analysis_Figures", "scores_most_difficult_part.png"
        )
    )
    plt.show()
