# visualization.py

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from data_prep import generate_dataloaders  # Replace 'data_prep' with the actual module name
import random
import os


def visualize_and_save_graph(graph_data, title="Graph Visualization", save_path="graph_visualization.png"):
    """
    Visualizes a graph from the GraphDataset and saves the visualization as an image.

    Parameters:
        graph_data (Data): Graph object from the dataset.
        title (str): Title for the plot.
        save_path (str): Path to save the visualization image.
    """
    # Convert PyTorch Geometric graph to NetworkX graph for visualization
    nx_graph = to_networkx(graph_data, edge_attrs=["edge_attr"], node_attrs=["x"])

    # Create plot
    plt.figure(figsize=(12, 8))
    pos = {}

    # Extract node positions from features (assuming 3D coordinates)
    for node, data in nx_graph.nodes(data=True):
        # The node attribute 'x' contains the (x, y, z) position
        pos[node] = data["x"][:2]  # Use (x, y) for 2D plotting .cpu().numpy()

    # Draw the graph
    edge_colors = [attr["edge_attr"].item() for _, _, attr in nx_graph.edges(data=True)]
    nx.draw_networkx_nodes(nx_graph, pos, node_size=50, node_color="blue")
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.viridis,
        width=2,
    )

    # Add color bar for edge weights
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)),
    )
    sm.set_array([])
    plt.colorbar(sm, label="Edge Features (e.g., Distance/Motion)")

    plt.title(title)
    plt.axis("off")

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Graph visualization saved as {save_path}")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    # Configuration
    config = {"batch_size": 1}  # Update batch size as needed

    # Load dataset and dataloaders
    graph_dataset, train_loader, val_loader, test_loader = generate_dataloaders(
        config["batch_size"]
    )

    # Sample a graph from the train loader
    random.seed(42)
    for batch in train_loader:
        # Assume the batch is a Data object or list of Data objects
        graph = batch[0] #batch if isinstance(batch, torch_geometric.data.Data) else batch[0]

        # Visualize and save the graph
        save_path = os.path.join(os.getcwd(), "graph_visualization.png")
        visualize_and_save_graph(graph, title="Graph Visualization from Real Dataset", save_path=save_path)
        break  # Only visualize the first graph
