import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from operator import itemgetter
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from SAD.model.tgat import TGAT
from SAD.utils import EarlyStopMonitor, get_neighbor_finder
from utils.utils import set_random_seed


@dataclass
class SADConfig:
    input_dim: int
    hidden_dim: int
    n_heads: int
    drop_out: float
    n_layer: int
    mode: str
    module_type: str
    anomaly_alpha: float
    supc_alpha: float
    memory_size: int
    sample_size: int
    n_neighbors: int


class SADTemporalDataset(Dataset):
    """Dataset wrapper that masks labels to the last timestamp for each node."""

    def __init__(
        self,
        full_data,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        indices: Sequence[int],
        n_layers: int,
        n_neighbors: int,
        double_labels: bool,
        dataset_name: str,
    ):
        super().__init__()
        self.full_data = full_data
        self.node_features = node_features.astype(np.float32)
        self.edge_features = edge_features.astype(np.float32)
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.double_labels = double_labels
        self.dataset_name = dataset_name.lower()

        raw = SimpleNamespace(
            sources=full_data.src_node_ids,
            destinations=full_data.dst_node_ids,
            timestamps=full_data.node_interact_times,
            edge_idxs=full_data.edge_ids,
        )
        self.ngh_finder = get_neighbor_finder(raw, uniform=False)
        self.samples = self._build_samples(indices)

    def _build_samples(self, indices: Sequence[int]) -> List[tuple]:
        samples = []
        if self.double_labels:
            labels_src, labels_dst = self.full_data.labels
            times_src, times_dst = self.full_data.labels_time
            for idx in indices:
                ts = self.full_data.node_interact_times[idx]
                # source node
                label = labels_src[idx]
                label_time = times_src[idx]
                effective_label = label if ts == label_time else -1
                if self.dataset_name in ["dsub", "dsub1m", "oag"] and label not in [0, 1]:
                    effective_label = -1
                samples.append((int(self.full_data.src_node_ids[idx]), float(ts), float(effective_label)))
                # destination node
                label = labels_dst[idx]
                label_time = times_dst[idx]
                effective_label = label if ts == label_time else -1
                if self.dataset_name in ["dsub", "dsub1m", "oag"] and label not in [0, 1]:
                    effective_label = -1
                samples.append((int(self.full_data.dst_node_ids[idx]), float(ts), float(effective_label)))
        else:
            labels = self.full_data.labels
            labels_time = self.full_data.labels_time
            for idx in indices:
                ts = self.full_data.node_interact_times[idx]
                label = labels[idx]
                label_time = labels_time[idx]
                effective_label = label if ts == label_time else -1
                if self.dataset_name in ["dsub", "dsub1m", "oag"] and label not in [0, 1]:
                    effective_label = -1
                samples.append((int(self.full_data.src_node_ids[idx]), float(ts), float(effective_label)))
        return samples

    @staticmethod
    def edge_padding(neigh_edge, neigh_time, edge_feat, src_neigh_idx, source_node):
        neigh_edge = np.concatenate((neigh_edge, np.tile(source_node.reshape(-1, 1), (1, 2))), axis=0)
        neigh_time = np.concatenate((neigh_time, np.zeros([1], dtype=neigh_time.dtype)), axis=0)
        edge_feat = np.concatenate((edge_feat, np.zeros([1, edge_feat.shape[1]], dtype=edge_feat.dtype)), axis=0)
        src_neigh_idx = np.concatenate((src_neigh_idx, np.zeros([1], dtype=src_neigh_idx.dtype)), axis=0)
        return neigh_edge, neigh_time, edge_feat, src_neigh_idx

    def __getitem__(self, item: int):
        center_node, current_time, label = self.samples[item]

        src_neigh_edge, src_neigh_time, src_neigh_idx = self.ngh_finder.get_temporal_neighbor_all(
            center_node, current_time, self.n_layers, self.n_neighbors
        )
        src_edge_feature = self.edge_features[src_neigh_idx].astype(np.float32)
        src_edge_to_time = current_time - src_neigh_time
        src_center_node_idx = np.reshape(np.array(center_node, dtype=np.int64), [-1])
        if src_neigh_edge.shape[0] == 0:
            src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx = self.edge_padding(
                src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx, src_center_node_idx
            )

        label = np.reshape(np.array(label, dtype=np.float32), [-1])
        current_time = np.reshape(np.array(current_time, dtype=np.float32), [-1])

        return {
            "src_center_node_idx": src_center_node_idx,
            "src_neigh_edge": torch.from_numpy(src_neigh_edge),
            "src_edge_feature": torch.from_numpy(src_edge_feature),
            "src_edge_to_time": torch.from_numpy(src_edge_to_time.astype(np.float32)),
            "init_edge_index": torch.from_numpy(src_neigh_idx),
            "current_time": torch.from_numpy(current_time),
            "label": torch.from_numpy(label),
        }

    def __len__(self):
        return len(self.samples)


class SADCollate:
    """Collate function that reindexes nodes per batch to avoid id collisions."""

    def __init__(self, node_features: np.ndarray):
        self.node_features = node_features.astype(np.float32)

    def reindex_fn(self, edge_list, center_node_idx, batch_idx):
        edge_list_projection = edge_list.view(-1).numpy().tolist()
        edge_list_projection = [str(x) for x in edge_list_projection]

        single_batch_idx = torch.unique(batch_idx).numpy().astype(np.int32).tolist()
        single_batch_idx = [str(x) for x in single_batch_idx]

        batch_idx_projection = batch_idx.reshape([-1, 1]).repeat((1, 2)).view(-1).numpy().astype(np.int32).tolist()
        batch_idx_projection = [str(x) for x in batch_idx_projection]

        center_node_idx_projection = center_node_idx.tolist()
        center_node_idx_projection = [str(x) for x in center_node_idx_projection]

        union_edge_list = list(map(lambda x: x[0] + "_" + x[1], zip(batch_idx_projection, edge_list_projection)))
        union_center_node_list = list(
            map(lambda x: x[0] + "_" + x[1], zip(single_batch_idx, center_node_idx_projection))
        )

        org_node_id = union_edge_list + union_center_node_list
        org_node_id = list(set(org_node_id))

        new_node_id = torch.arange(0, len(org_node_id)).numpy()
        reid_map = dict(zip(org_node_id, new_node_id))
        true_org_node_id = [int(x.split("_")[1]) for x in org_node_id]
        true_org_node_id = np.array(true_org_node_id)

        keys = union_edge_list
        new_edge_list = itemgetter(*keys)(reid_map)
        new_edge_list = np.array(new_edge_list).reshape([-1, 2])
        new_edge_list = torch.from_numpy(new_edge_list)
        batch_node_features = self.node_features[true_org_node_id]
        new_center_node_idx = np.array(itemgetter(*union_center_node_list)(reid_map))

        return new_center_node_idx, new_edge_list, batch_node_features

    @staticmethod
    def get_batchidx_fn(edge_list):
        batch_size = len(edge_list)
        feat_max_len = np.sum([feat.shape[0] for feat in edge_list])

        mask = torch.zeros((feat_max_len))

        count = 0
        for i, ifeat in enumerate(edge_list):
            size = ifeat.shape[0]
            mask[count : count + size] = i + 1
            count += size
        return mask

    def __call__(self, batch):
        src_edge_feat = torch.cat([b["src_edge_feature"] for b in batch], dim=0)
        src_edge_to_time = torch.cat([b["src_edge_to_time"] for b in batch], dim=0)

        init_edge_index = torch.cat([b["init_edge_index"] for b in batch], dim=0)

        src_center_node_idx = np.concatenate([b["src_center_node_idx"] for b in batch], axis=0)

        batch_idx = self.get_batchidx_fn([b["src_neigh_edge"] for b in batch])
        src_neigh_edge = torch.cat([b["src_neigh_edge"] for b in batch], dim=0)
        src_center_node_idx, src_neigh_edge, src_node_features = self.reindex_fn(
            src_neigh_edge, src_center_node_idx, batch_idx
        )

        label = torch.cat([b["label"] for b in batch], dim=0)
        current_time = torch.cat([b["current_time"] for b in batch], dim=0)

        return {
            "src_edge_feat": src_edge_feat,
            "src_edge_to_time": src_edge_to_time,
            "src_center_node_idx": torch.from_numpy(src_center_node_idx),
            "src_neigh_edge": src_neigh_edge,
            "src_node_features": torch.from_numpy(src_node_features),
            "init_edge_index": init_edge_index,
            "batch_idx": batch_idx,
            "current_time": current_time,
            "labels": label,
        }


def sad_criterion(prediction_dict, labels, model, config: SADConfig, device):
    mask = labels > -1
    if mask.sum() == 0:
        zero = torch.zeros(1, device=device, requires_grad=True)
        return zero, zero, zero, zero

    filtered = {
        key: value[mask] for key, value in prediction_dict.items() if key not in ["root_embedding", "group", "dev"]
    }
    labels = labels[mask]
    logits = filtered["logits"]

    loss_classify = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    loss = loss_classify.clone()

    loss_anomaly = torch.tensor(0.0, device=device)
    loss_supc = torch.tensor(0.0, device=device)

    if config.mode in ["sad", "gdn"]:
        loss_anomaly = model.gdn.dev_loss(
            torch.squeeze(labels), torch.squeeze(filtered["anom_score"]), torch.squeeze(filtered["time"])
        )
        loss = loss + config.anomaly_alpha * loss_anomaly
        if config.mode == "sad":
            loss_supc = model.suploss(prediction_dict["root_embedding"], prediction_dict["group"], prediction_dict["dev"])
            loss = loss + config.supc_alpha * loss_supc

    return loss, loss_classify, loss_anomaly, loss_supc


def sad_eval_epoch(data_loader, model, config: SADConfig, device, dataset_name: str = ""):
    loss_values, all_probs, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_sample in data_loader:
            x = model(
                batch_sample["src_edge_feat"].to(device),
                batch_sample["src_edge_to_time"].to(device),
                batch_sample["src_center_node_idx"].to(device),
                batch_sample["src_neigh_edge"].to(device),
                batch_sample["src_node_features"].to(device),
                batch_sample["current_time"].to(device),
                batch_sample["labels"].to(device),
            )
            y = batch_sample["labels"].to(device)
            mask = y > -1
            if mask.sum() == 0:
                continue

            loss, loss_classify, _, _ = sad_criterion(x, y, model, config, device)
            loss_values.append(loss_classify.item())

            probs = x["logits"][mask].sigmoid().detach().cpu()
            labels = y[mask].detach().cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    if len(all_labels) == 0:
        return {"roc_auc": 0.0, "acc": 0.0, "loss": 0.0}

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    labels_np = labels.numpy()
    probs_np = probs.numpy()
    unique_labels = np.unique(labels_np)
    acc = accuracy_score(labels_np, (probs_np >= 0.5).astype(int))

    roc_auc = 0.0
    if len(unique_labels) > 1 and set(unique_labels).issubset({0, 1}) and dataset_name.lower() not in ["oag"]:
        roc_auc = roc_auc_score(labels_np, probs_np)

    loss_value = float(np.mean(loss_values)) if loss_values else 0.0
    return {"roc_auc": roc_auc, "acc": acc, "loss": loss_value}


def run_sad(args, data):
    best_test_all = [0.0, 0.0]
    double_way_datasets = args.double_way_datasets
    if torch.cuda.is_available():
        visible_devices = torch.cuda.device_count()
        target_idx = min(args.gpu, max(visible_devices - 1, 0))
        device = torch.device(f"cuda:{target_idx}")
    else:
        device = torch.device("cpu")

    node_feat_dim = data["node_raw_features"].shape[1]
    config = SADConfig(
        input_dim=node_feat_dim,
        hidden_dim=args.sad_hidden_dim,
        n_heads=args.sad_num_heads,
        drop_out=args.sad_dropout,
        n_layer=args.sad_num_layers,
        mode=args.sad_mode,
        module_type=args.sad_module_type,
        anomaly_alpha=args.sad_anomaly_alpha,
        supc_alpha=args.sad_supc_alpha,
        memory_size=args.sad_memory_size,
        sample_size=args.sad_sample_size,
        n_neighbors=args.num_neighbors,
    )

    batch_size = args.sad_batch_size if args.sad_batch_size is not None else args.batch_size
    learning_rate = args.sad_learning_rate if args.sad_learning_rate is not None else args.learning_rate

    full_data = data["full_data"]
    total_interactions = len(full_data.src_node_ids)
    train_indices = np.arange(0, data["val_offest"])
    val_indices = np.arange(data["val_offest"], data["test_offest"])
    test_indices = np.arange(data["test_offest"], total_interactions)

    double_labels = data["dataset_name"] in double_way_datasets

    for run in range(args.start_runs, args.end_runs):
        set_random_seed(seed=run)
        args.seed = run

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_folder = f"./logs/sad/{args.prefix}/{args.dataset_name}/seed_{args.seed}/"
        os.makedirs(log_folder, exist_ok=True)
        fh = logging.FileHandler(f"{log_folder}{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f"configuration is {args}")

        train_dataset = SADTemporalDataset(
            full_data=full_data,
            node_features=data["node_raw_features"],
            edge_features=data["edge_raw_features"],
            indices=train_indices,
            n_layers=config.n_layer,
            n_neighbors=config.n_neighbors,
            double_labels=double_labels,
            dataset_name=data["dataset_name"],
        )
        val_dataset = SADTemporalDataset(
            full_data=full_data,
            node_features=data["node_raw_features"],
            edge_features=data["edge_raw_features"],
            indices=val_indices,
            n_layers=config.n_layer,
            n_neighbors=config.n_neighbors,
            double_labels=double_labels,
            dataset_name=data["dataset_name"],
        )
        test_dataset = SADTemporalDataset(
            full_data=full_data,
            node_features=data["node_raw_features"],
            edge_features=data["edge_raw_features"],
            indices=test_indices,
            n_layers=config.n_layer,
            n_neighbors=config.n_neighbors,
            double_labels=double_labels,
            dataset_name=data["dataset_name"],
        )

        collate = SADCollate(node_features=data["node_raw_features"])

        loader_train = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )
        loader_valid = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )
        loader_test = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )

        model = TGAT(config, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        best_val_auc = -1.0
        best_test_metrics = {"roc_auc": 0.0, "acc": 0.0}

        for epoch in range(args.num_epochs_sad):
            model.train()
            running_losses, running_class_losses, running_anomaly_losses, running_sup_losses = [], [], [], []
            with tqdm(total=len(loader_train), ncols=120) as t:
                for batch_sample in loader_train:
                    t.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()
                    x = model(
                        batch_sample["src_edge_feat"].to(device),
                        batch_sample["src_edge_to_time"].to(device),
                        batch_sample["src_center_node_idx"].to(device),
                        batch_sample["src_neigh_edge"].to(device),
                        batch_sample["src_node_features"].to(device),
                        batch_sample["current_time"].to(device),
                        batch_sample["labels"].to(device),
                    )
                    y = batch_sample["labels"].to(device)
                    if (y > -1).sum() == 0:
                        t.set_postfix(skip="no_valid_labels")
                        t.update(1)
                        continue
                    loss, loss_classify, loss_anomaly, loss_supc = sad_criterion(x, y, model, config, device)
                    if not loss.requires_grad:
                        t.set_postfix(skip="no_grad_loss")
                        t.update(1)
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()

                    running_losses.append(loss.item())
                    running_class_losses.append(loss_classify.item() if torch.is_tensor(loss_classify) else 0.0)
                    running_anomaly_losses.append(loss_anomaly.item() if torch.is_tensor(loss_anomaly) else 0.0)
                    running_sup_losses.append(loss_supc.item() if torch.is_tensor(loss_supc) else 0.0)
                    t.set_postfix(
                        loss=np.mean(running_class_losses) if running_class_losses else 0.0,
                        loss_anom=np.mean(running_anomaly_losses) if running_anomaly_losses else 0.0,
                        loss_sup=np.mean(running_sup_losses) if running_sup_losses else 0.0,
                    )
                    t.update(1)

            val_metrics = sad_eval_epoch(loader_valid, model, config, device, data["dataset_name"])
            test_metrics = sad_eval_epoch(loader_test, model, config, device, data["dataset_name"])

            logger.info(
                f"epoch {epoch} train loss {np.mean(running_losses) if running_losses else 0.0:.4f} | "
                f"val auc {val_metrics['roc_auc']:.4f} acc {val_metrics['acc']:.4f} | "
                f"test auc {test_metrics['roc_auc']:.4f} acc {test_metrics['acc']:.4f}"
            )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_test_metrics = test_metrics

            if early_stopper.early_stop_check(val_metrics["roc_auc"]):
                logger.info(f"No improvement for {early_stopper.max_round} epochs, stop training.")
                break

        logger.info(
            f"Best test metrics at val-best epoch, auc: {best_test_metrics['roc_auc']:.4f}, "
            f"acc: {best_test_metrics['acc']:.4f}"
        )
        best_test_all = [best_test_metrics["roc_auc"], best_test_metrics["acc"]]

        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    return best_test_all
