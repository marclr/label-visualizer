import csv
import argparse
import pandas as pd
import umap
import hdbscan
import holoviews as hv
from sentence_transformers import SentenceTransformer

hv.extension('bokeh')


def get_labels(input_file):
    """Extract unique labels from the CSV file."""
    labels = []
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tags = row['tags'].lower().split(',')
            labels.extend(tags)
    return list(set(labels))


def plot_label_clusters(labels):
    """
    Process labels using embeddings, UMAP, HDBSCAN, and generate an interactive scatter plot
    with HoloViews.
    """
    # 1. Embeddings with Sentence-BERT
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(labels)

    # 2. Dimensionality reduction with UMAP
    reducer = umap.UMAP(n_neighbors=10, n_components=2, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)

    # 3. Clustering with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusters = clusterer.fit_predict(embedding_2d)

    # 4. Create DataFrame with the results
    df = pd.DataFrame(embedding_2d, columns=["x", "y"])
    df["label"] = pd.Categorical(labels)
    df["cluster"] = pd.Categorical(clusters).codes
    num_clusters = len(df['cluster'].unique())
    print(f"Number of clusters: {num_clusters}")

    # 5. Visualization with HoloViews
    scatter = hv.Scatter(df, kdims=['x', 'y'], vdims=['cluster', 'label']).opts(
        width=800, height=600,
        color='cluster', cmap='Category20', size=5,
        tools=['hover'], alpha=0.8
    )

    return scatter


def main():
    parser = argparse.ArgumentParser(description='Generate interactive label visualization.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument(
        'output_file', type=str, nargs='?', default='plot.html',
        help='Path to save the output HTML file (default: plot.html)'
    )
    args = parser.parse_args()

    if not args.input_file:
        print("Error: You must provide an input file.")
        parser.print_help()
        return

    labels = get_labels(args.input_file)
    print(f"Total unique labels: {len(labels)}")

    plot = plot_label_clusters(labels)
    hv.save(plot, args.output_file)
    print(f"Visualization saved as '{args.output_file}'")


if __name__ == '__main__':
    main()
