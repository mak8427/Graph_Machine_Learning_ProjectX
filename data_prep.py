import os
import numpy as np
from hik.data.scene import Scene
from einops import rearrange
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

body_edges = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            (16, 13),
            (17, 14),
            (13, 12),
            (14, 12),
            (12, 15),
            (15, 23),
            (24, 25),
            (24, 26),
            (25, 27),
            (26, 28),
        ]


class HiKDataset(Dataset):
    def __init__(self, datasets, data_location, sample_length=30, step_size=30, downsample_rate=6):
        self.poses = []
        self.activities = []

        for dataset in datasets:
            scene = Scene.load_from_paths(
                dataset,
                os.path.join(data_location, "poses"),
                os.path.join(data_location, "scenes"),
                os.path.join(data_location, "body_models")
            )
            splits = scene.get_splits(sample_length, stepsize=step_size)
            self.process_data(splits, downsample_rate)

        # Combine all datasets into single arrays
        self.poses = np.concatenate(self.poses, axis=0)
        self.activities = np.concatenate(self.activities, axis=0)

    def process_data(self, splits, downsample_rate):
        poses3d = splits["poses3d"]
        masks = splits["masks"]
        activities = splits["activities"]

        n_seq, n_frames, n_person, n_joints, n_xyz = poses3d.shape

        poses = rearrange(
            poses3d, "n_seq n_frames n_person n_joints n_xyz -> (n_seq n_person) n_frames n_xyz n_joints"
        )
        masks = rearrange(
            masks, "n_seq n_frames n_person -> (n_seq n_person) n_frames"
        )
        activities = rearrange(
            activities, "n_seq n_frames n_person n_activities -> (n_seq n_person) n_frames n_activities"
        )

        # Filter sequences with missing frames
        valid_indices = np.all(masks, axis=1)
        poses = poses[valid_indices]
        activities = activities[valid_indices]

        # Downsample frames
        poses = poses[:, ::downsample_rate, :, :]
        activities = activities[:, ::downsample_rate, :]

        # Append to global dataset
        self.poses.append(poses)
        self.activities.append(activities)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        return {
            "poses": torch.tensor(self.poses[idx], dtype=torch.float32),
            "activity_label": torch.tensor(self.activities[idx, self.activities.shape[1] // 2], dtype=torch.float32)
        }

    def save(self, save_path):
        """
        Save the processed dataset to a specified path.
        Args:
            save_path (str): Directory where the dataset will be saved.
        """
        os.makedirs(save_path, exist_ok=True)

        # Calculate the middle frame index
        middle_frame_idx = self.activities.shape[1] // 2

        # Save poses as-is
        torch.save(torch.tensor(self.poses, dtype=torch.float32), os.path.join(save_path, "poses.pt"))

        # Extract activities for the middle frame and save
        middle_activities = self.activities[:, middle_frame_idx, :]
        torch.save(torch.tensor(middle_activities, dtype=torch.float32), os.path.join(save_path, "activities.pt"))

        print(f"Dataset saved to {save_path}")






class GraphDataset(Dataset):
    def __init__(self, body_edges, sit_stand_only=False, main_training=True):
        """
        Args:
            body_edges (list of tuple): List of body edges defining spatial connections.
            sit_stand_only (bool): Whether to filter only sitting, standing, and other categories.
        """
        self.main_training = main_training
        self.graphs = []
        self.data_path = 'data_processed'
        self.body_edges = body_edges

        # Load tensors from the saved files
        self.poses = torch.load(os.path.join(self.data_path, "poses.pt"), weights_only=True)
        self.activities = torch.load(os.path.join(self.data_path, "activities.pt"), weights_only=True)  # Shape: (N, 82)

        # self.poses = self.poses[:512]
        # self.activities = self.activities[:512]

        if self.poses.shape[0] != self.activities.shape[0]:
            raise ValueError("Mismatch in number of samples between poses and activities")

        # Filter out sequences with sitting, standing, or neither, if requested
        if sit_stand_only:
            self.filter_sit_stand_other()

        # if sit_stand_only:
            # self.poses = self.poses[:int(len(self.poses) * 0.2)]
            # self.activities = self.activities[:int(len(self.activities) * 0.2)]
        
        # if main_training:
            # self.poses = self.poses[int(len(self.poses) * 0.2):]
            # self.activities = self.activities[int(len(self.activities) * 0.2):]

        for idx in range(len(self.poses)):
            pose = self.poses[idx]
            label = self.activities[idx]
            self.graphs.append(self.create_graph(pose, label))

    def create_graph(self, poses, label):
        n_frames, n_xyz, n_joints = poses.shape

        # Flatten poses for node features
        features = poses.view(-1, n_xyz)

        edges = []
        edge_features = []

        # Add spatial edges and compute spatial edge features
        for t in range(n_frames):
            for j1, j2 in self.body_edges:
                node_id1 = t * n_joints + j1
                node_id2 = t * n_joints + j2
                edges.append([node_id1, node_id2])
                edges.append([node_id2, node_id1])

                # Compute Euclidean distance for spatial edge feature
                p1 = poses[t, :, j1]
                p2 = poses[t, :, j2]
                distance = torch.norm(p1 - p2)
                edge_features.append(distance)
                edge_features.append(distance)

        # Add temporal edges and compute temporal edge features
        for t in range(n_frames - 1):
            for j in range(n_joints):
                node_id1 = t * n_joints + j
                node_id2 = (t + 1) * n_joints + j
                edges.append([node_id1, node_id2])
                edges.append([node_id2, node_id1])

                # Compute motion for temporal edge feature (velocity vector norm)
                p1 = poses[t, :, j]
                p2 = poses[t + 1, :, j]
                motion = torch.norm(p2 - p1)
                edge_features.append(motion)
                edge_features.append(motion)

        # Convert edges and edge features to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float).unsqueeze(-1)

        # Create graph data object
        graph_data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=label.unsqueeze(0),
            num_nodes=n_frames * n_joints
        )

        return graph_data

    def filter_sit_stand_other(self):
        """
        Filter labels to keep only sitting, standing, and other activities.
        Reassign labels:
        - 0: Sitting (58th entry in activities is True)
        - 1: Standing (61st entry in activities is True)
        - 2: Neither sitting nor standing
        """
        filtered_poses = []
        filtered_activities = []

        for pose, activity in zip(self.poses, self.activities):
            if activity[58]:  # Sitting
                new_label = 0
            elif activity[61]:  # Standing
                new_label = 1
            else:  # Neither
                new_label = 2

            # Append the filtered data
            filtered_poses.append(pose)
            filtered_activities.append(new_label)

        # Convert filtered lists to tensors
        self.poses = torch.stack(filtered_poses)
        self.activities = torch.tensor(filtered_activities, dtype=torch.long)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def save(self, save_path):
        """
        Save the processed dataset to a specified path.
        Args:
            save_path (str): Directory where the dataset will be saved.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save the processed graphs
        torch.save(self.graphs, os.path.join(save_path, "graphs.pt"))

        print(f"Dataset saved to {save_path}")


def normalize_batch(batch):
    """Normalize the x, y, z coordinates in a batch."""
    all_features = torch.cat([data.x for data in batch], dim=0)
    
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)

    for data in batch:
        data.x = (data.x - mean) / (std + 1e-8)  # Avoid division by zero

    return batch

def collate_fn(batch):
    """Custom collate function to normalize and prepare batches."""
    normalized_batch = normalize_batch(batch)
    return DataLoader.collate(normalized_batch)

def generate_dataloaders_sit_stand(batch_size):
    data_location = 'data'
    dataset = 'A'
    sample_length = 30
    step_size = 30
    # hik_dataset = HiKDataset(dataset, data_location, sample_length, step_size)
    # graph_dataset = GraphDataset(hik_dataset, body_edges)
    # graph_dataset = GraphDataset(body_edges, sit_stand_only=True, main_training=False)
    graph_dataset = torch.load('data_processed/graphs.pt', weights_only=True)
    graph_dataset = graph_dataset[:int(len(graph_dataset) * 0.2)]

    train_size = int(0.8 * len(graph_dataset))
    val_size = int(0.1 * len(graph_dataset))
    test_size = len(graph_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        graph_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return graph_dataset, train_loader, val_loader, test_loader

def collate_fn_train(batch):
    # drop the 58th and 61st entry in the activities
    for data in batch:
        data.y = data.y[:, [i for i in range(82) if i != 58 and i != 61]]
    return DataLoader.collate(batch)

def generate_dataloaders(batch_size):
    data_location = 'data'
    dataset = 'A'
    sample_length = 30
    step_size = 30
    # hik_dataset = HiKDataset(dataset, data_location, sample_length, step_size)
    # graph_dataset = GraphDataset(hik_dataset, body_edges)
    # graph_dataset = GraphDataset(body_edges)
    graph_dataset = torch.load('data_processed/graphs.pt')
    graph_dataset = graph_dataset[int(len(graph_dataset) * 0.2):]

    # # undersample the dataset by throwing away 50% of the entries with label 58 and 61 present
    # filtered_graph_dataset = []
    # for data in graph_dataset:
    #     if data.y[0][58] or data.y[0][61]:
    #         # print(data.y[0][58], data.y[0][61])
    #         if np.random.rand() < 0.5:
    #             continue
    #     filtered_graph_dataset.append(data)

    # print(f"Filtered dataset size: {len(filtered_graph_dataset)}")

    # graph_dataset = filtered_graph_dataset
    train_size = int(0.8 * len(graph_dataset))
    val_size = int(0.1 * len(graph_dataset))
    test_size = len(graph_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        graph_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return graph_dataset, train_loader, val_loader, test_loader

if __name__=='__main__':
    # Initialize the dataset
    # datasets = ['A', 'B', 'C', 'D']
    # data_location = 'data'
    # hik_dataset = HiKDataset(datasets, data_location, sample_length=30, step_size=30, downsample_rate=6)

    # Save the dataset
    # save_path = "data_processed"
    # hik_dataset.save(save_path)
    graph_dataset = GraphDataset(body_edges)
    # save the graph_dataset
    save_path = "data_processed"
    graph_dataset.save(save_path)
