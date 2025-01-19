import os
import numpy as np
from hik.data.scene import Scene
from einops import rearrange
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data

class HiKDataset(Dataset):
    def __init__(self, dataset, data_location, sample_length=30, step_size=30, downsample_rate=6):
        self.scene = Scene.load_from_paths(
            dataset,
            os.path.join(data_location, "poses"),
            os.path.join(data_location, "scenes"),
            os.path.join(data_location, "body_models")
        )
        
        self.splits = self.scene.get_splits(sample_length, stepsize=step_size)
        self.process_data(downsample_rate)

    def process_data(self, downsample_rate):
        poses3d = self.splits["poses3d"]
        masks = self.splits["masks"]
        activities = self.splits["activities"]

        n_seq, n_frames, n_person, n_joints, n_xyz = poses3d.shape
        self.poses = rearrange(
            poses3d, "n_seq n_frames n_person n_joints n_xyz -> (n_seq n_person) n_frames n_xyz n_joints"
        )
        self.masks = rearrange(
            masks, "n_seq n_frames n_person -> (n_seq n_person) n_frames"
        )
        self.activities = rearrange(
            activities, "n_seq n_frames n_person n_activities -> (n_seq n_person) n_frames n_activities"
        )

        valid_indices = np.all(self.masks, axis=1)

        self.poses = self.poses[valid_indices]
        self.activities = self.activities[valid_indices]

        self.poses = self.poses[:, ::downsample_rate, :, :]
        self.activities = self.activities[:, ::downsample_rate, :]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return {
            "poses": torch.tensor(self.poses[idx], dtype=torch.float32),
            "activity_label": torch.tensor(self.activities[idx, self.activities.shape[1] // 2], dtype=torch.float32)
        }

class GraphDataset(Dataset):
    def __init__(self, hik_dataset):
        self.graphs = []
        for idx in range(len(hik_dataset)):
            data = hik_dataset[idx]
            poses = data["poses"]
            label = data["activity_label"]

            graph_data = self.create_graph(poses, label)
            self.graphs.append(graph_data)

    def create_graph(self, poses, label):
        n_frames, n_xyz, n_joints = poses.shape

        features = poses.view(-1, n_xyz)
        edges = []

        for t in range(n_frames):
            for j in range(n_joints - 1):
                node_id = t * n_joints + j
                next_node_id = t * n_joints + (j + 1)
                edges.append([node_id, next_node_id])
                edges.append([next_node_id, node_id])

        for t in range(n_frames - 1):
            for j in range(n_joints):
                node_id = t * n_joints + j
                next_node_id = (t + 1) * n_joints + j
                edges.append([node_id, next_node_id])
                edges.append([next_node_id, node_id])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        graph_data = Data(
            x=features,
            edge_index=edge_index,
            y=label.unsqueeze(0)
        )

        return graph_data

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]