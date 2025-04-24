from cyvcf2 import VCF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def get_vcf_data(vcf_file):
    vcf_reader = vcf.Reader(open(vcf_file, 'r'))
    data = []
    for record in vcf_reader:
        data.append([record.CHROM, record.POS, record.REF, record.ALT[0], record.INFO['AF'][0]])
    return pd.DataFrame(data, columns=['CHROM', 'POS', 'REF', 'ALT'])

def read_all_vcfs(directory):
    vcf_files = [f for f in os.listdir(directory) if f.endswith('.vcf')]
    return vcf_files

def plot_overlap_heatmap(directory, plot_title):
    variant_data = {}

    vcfs = read_all_vcfs(directory)
    # Parse each VCF file
    for i in range(1, 12):
        vcf_reader = VCF(os.path.join(directory, vcfs[i-1]))
        for record in vcf_reader:
            variant_key = f'{record.CHROM}_{record.POS}_{record.REF}_{record.ALT[0]}'
            if variant_key not in variant_data:
                variant_data[variant_key] = [0] * 11  # initialize with 0s for each file
            variant_data[variant_key][i-1] = 1  # set 1 for presence of the variant

    # Convert the dictionary to a pandas DataFrame
    variant_matrix = pd.DataFrame.from_dict(variant_data, orient='index', columns=[f'File{i}' for i in range(1, 12)])

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(variant_matrix, cmap='viridis', cbar_kws={'label': 'Presence of Variant'})
    plt.title(plot_title)
    plt.ylabel('Variant')
    plt.xlabel('VCF File')
    plt.show()

def plot_jaccard_similarity(directory, plt_title):
    variant_data = {}

    vcfs = read_all_vcfs(directory)
    # Parse each VCF file
    for i in range(1, 12):
        vcf_reader = VCF(os.path.join(directory, vcfs[i-1]))
        for record in vcf_reader:
            variant_key = f'{record.CHROM}_{record.POS}_{record.REF}_{record.ALT[0]}'
            if variant_key not in variant_data:
                variant_data[variant_key] = [0] * 11  # initialize with 0s for each file
            variant_data[variant_key][i-1] = 1  # set 1 for presence of the variant

    # Convert the dictionary to a pandas DataFrame
    variant_matrix = pd.DataFrame.from_dict(variant_data, orient='index', columns=[f'File{i}' for i in range(1, 12)])

    # Calculate Jaccard Similarity
    jaccard_similarity = pd.DataFrame(np.zeros((11, 11)), index=variant_matrix.columns, columns=variant_matrix.columns)

    for i in range(11):
        for j in range(i, 11):
            intersection = np.logical_and(variant_matrix.iloc[:, i], variant_matrix.iloc[:, j]).sum()
            union = np.logical_or(variant_matrix.iloc[:, i], variant_matrix.iloc[:, j]).sum()
            similarity = intersection / union if union != 0 else 0
            jaccard_similarity.iloc[i, j] = similarity
            jaccard_similarity.iloc[j, i] = similarity  # the matrix is symmetric

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard_similarity, annot=True, cmap='viridis', cbar_kws={'label': 'Jaccard Similarity'})
    plt.title(plt_title)
    plt.savefig(os.path.join("Figures", "".join(plt_title)+'.png'))
    plt.show()

def parse_vcf(file_path):
    """
    Parse a VCF file and return a set of unique variant identifiers (chromosome, position, reference, alternate).
    """
    vcf_reader = VCF(file_path)
    variants = set()
    for record in vcf_reader:
        variant_id = (record.CHROM, record.POS, record.REF, tuple(record.ALT))
        variants.add(variant_id)
    return variants


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def plot_jaccard_similarity_matrix(vcfs, plt_title):
    # Parse all VCFs
    vcf_variants = [parse_vcf(file_path) for file_path in vcfs]

    # Calculate Jaccard similarity matrix
    similarity_matrix = np.zeros((len(vcfs), len(vcfs)))
    for i in range(len(vcfs)):
        for j in range(len(vcfs)):
            similarity_matrix[i][j] = jaccard_similarity(vcf_variants[i], vcf_variants[j])

 # Convert similarity matrix to distance matrix for clustering
    distance_matrix = 1 - similarity_matrix

    # Using seaborn's clustermap to plot the heatmap with dendrograms
    sns.clustermap(distance_matrix, cmap='coolwarm', xticklabels=[os.path.basename(fp) for fp in vcfs], yticklabels=[os.path.basename(fp) for fp in vcfs], figsize=(30, 30), method='average')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title('Jaccard Distance and Dendrogram between VCF Files', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join("Figures", "".join(plt_title)+'.png'))
    plt.show()



def calculate_metrics(test_variants, true_variants):
    """
    Calculate precision, recall, F1 score, and accuracy.
    """
    true_positives = len(test_variants & true_variants)
    false_positives = len(test_variants - true_variants)
    false_negatives = len(true_variants - test_variants)
    true_negatives = 0  # True negatives are not directly applicable without knowing the total number of possible variants.

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    # Accuracy is not well-defined for this task without knowing the total number of non-variants (true negatives)


    return precision, recall, f1, accuracy, true_positives, false_positives, false_negatives, true_negatives


def plot_pca(vcf_files, plt_title):
    # Identify unique variants across all files
    all_variants = set()
    for file_path in vcf_files:
        all_variants |= parse_vcf(file_path)

    # Sort variants to ensure consistent ordering
    all_variants = sorted(list(all_variants))

    # Create a binary matrix for presence/absence of variants in each file
    binary_matrix = np.zeros((len(vcf_files), len(all_variants)), dtype=int)
    for i, file_path in enumerate(vcf_files):
        file_variants = parse_vcf(file_path)
        for j, variant in enumerate(all_variants):
            if variant in file_variants:
                binary_matrix[i, j] = 1

    # Perform PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(binary_matrix)
    # Extract explained variance ratio
    explained_variance = pca.explained_variance_ratio_


    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
    colors = {"ss": "r", "strelka": "g", "mutect": "b"}
    markers = {"bwa": "o", "bowtie": "s"}

    for i, file_path in enumerate(vcf_files):
        if "bed" not in file_path:
            c = colors[file_path.split("/")[-1].split('_')[1]]
            m = markers[file_path.split("/")[-1].split('_')[3][:-4]]
            plt.scatter(principalComponents[i, 0], principalComponents[i, 1], label=file_path.split('/')[-1][:-4], c=c, marker=m, alpha=0.5, s=100)
        else:
            plt.scatter(principalComponents[i, 0], principalComponents[i, 1], label=file_path.split('/')[-1][:-4], s=150, c='k', marker='x', alpha=0.8)
    
    # access legend objects automatically created from data
    handles, labels = plt.gca().get_legend_handles_labels()

    # create manual symbols for legend
    ss_bwa = Line2D([0], [0], label='SS BWA', marker='o', markersize=10, 
            c='r', linestyle="")
    ss_bowtie = Line2D([0], [0], label='SS Bowtie', marker='s', markersize=10,
            c='r',  linestyle="")
    strelka_bwa = Line2D([0], [0], label='Strelka BWA', marker='o', markersize=10,
            c='g',linestyle="")
    strelka_bowtie = Line2D([0], [0], label='Strelka Bowtie', marker='s', markersize=10,
            c='g',  linestyle="")
    mutect_bwa = Line2D([0], [0], label='Mutect BWA', marker='o', markersize=10,
            c='b', linestyle="")
    mutect_bowtie = Line2D([0], [0], label='Mutect Bowtie', marker='s', markersize=10,
            c='b', linestyle="")
    bed = Line2D([0], [0], label='High Confidence', marker='x', markersize=10, color='k', linestyle="")

    # add legend to plot
    plt.legend(handles=[ss_bwa, ss_bowtie, strelka_bwa, strelka_bowtie, mutect_bwa, mutect_bowtie, bed], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)', fontsize=14)
    plt.title('PCA of VCF Files and High-Confindence Variants', fontsize=16)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig(os.path.join("Figures", "".join(plt_title)+'.pdf'))
    plt.show()