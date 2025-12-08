"""Feature extractors for SELEX PoC.

This module is Linux-only due to ViennaRNA dependency.
"""

import inspect
import itertools
from typing import Literal

import cupy
import numpy as np
import torch
from multimolecule import RnaFmModel, RnaTokenizer
from torch.nn import Linear
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from ViennaRNA import RNA

from selex_analyze_poc.base import FeatureExtractor


def _get_array_module(device: str):
    """Get numpy or cupy depending on device."""
    if device == "cuda":
        return cupy
    return np


class OneHotExtractor(FeatureExtractor):
    """One-hot encodes RNA sequences.

    Each sequence is converted to an (L, 4) array, where L is the sequence
    length. The arrays are then flattened to create a fixed-size vector.

    Args:
        max_length: Sequences will be padded or truncated to this length.
        device: "cpu" or "cuda".
    """

    def __init__(self, max_length: int = 100, device: str = "cpu"):
        super().__init__()
        self.max_length = max_length
        self.alphabet = "ACGU"
        self.char_to_int = {char: i for i, char in enumerate(self.alphabet)}
        self.device = device

    def extract(self, sequences: list[str]) -> np.ndarray:
        """One-hot encode a list of sequences.

        Args:
            sequences: List of RNA sequences.

        Returns:
            A 2D numpy or cupy array of shape (num_sequences, max_length * 4).
        """
        self._last_extracted_sequences = sequences
        xp = _get_array_module(self.device)
        encoded_sequences = xp.zeros(
            (len(sequences), self.max_length, len(self.alphabet))
        )

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_length)
            for j, char in enumerate(seq[:seq_len]):
                if char in self.char_to_int:
                    encoded_sequences[i, j, self.char_to_int[char]] = 1

        return encoded_sequences.reshape(len(sequences), -1)

    @property
    def dim(self) -> int:
        """Dimension of the one-hot encoded vector."""
        return self.max_length * len(self.alphabet)

    @property
    def name(self) -> str:
        return f"OneHot (L={self.max_length})"


class ThermoExtractor(FeatureExtractor):
    """Extracts thermodynamic features using ViennaRNA.

    Features extracted:
    - Minimum Free Energy (MFE)
    - Ensemble free energy
    - Flattened Base Pair Probability Matrix (BPPM)
    """

    def __init__(self, max_length: int = 100, device: str = "cpu"):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self._dim = 2 + (self.max_length * self.max_length)

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extracts thermodynamic features."""
        self._last_extracted_sequences = sequences
        xp = _get_array_module(self.device)

        all_features = xp.zeros((len(sequences), self.dim))

        for i, seq in enumerate(sequences):
            seq = seq[: self.max_length]

            fc = RNA.fold_compound(seq)

            (_mfe_struct, mfe) = fc.mfe()
            ens_ene = fc.pf()[1]

            bppm = fc.bpp()
            bppm_matrix = xp.asarray(bppm)
            bppm_matrix = bppm_matrix[1:, 1:]

            padded_bppm = xp.zeros((self.max_length, self.max_length))
            seq_len = len(seq)
            if seq_len > 0:
                padded_bppm[:seq_len, :seq_len] = bppm_matrix

            all_features[i, 0] = mfe
            all_features[i, 1] = ens_ene
            all_features[i, 2:] = padded_bppm.flatten()

        return all_features

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        return self._dim

    @property
    def name(self) -> str:
        return "Thermodynamic (ViennaRNA)"


class KmerExtractor(FeatureExtractor):
    """Counts k-mer frequencies in RNA sequences.

    Args:
        k: The length of the k-mers.
        device: "cpu" or "cuda".
    """

    def __init__(self, k: int = 3, device: str = "cpu"):
        super().__init__()
        if not k > 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.alphabet = "ACGU"
        self.kmers = ["".join(p) for p in itertools.product(self.alphabet, repeat=k)]
        self.kmer_to_int = {kmer: i for i, kmer in enumerate(self.kmers)}
        self.device = device

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Count k-mers in a list of sequences.

        Args:
            sequences: List of RNA sequences.

        Returns:
            A 2D numpy or cupy array of shape (num_sequences, 4**k).
        """
        self._last_extracted_sequences = sequences
        xp = _get_array_module(self.device)
        counts = xp.zeros((len(sequences), len(self.kmers)))
        for i, seq in enumerate(sequences):
            for j in range(len(seq) - self.k + 1):
                kmer = seq[j : j + self.k]
                if kmer in self.kmer_to_int:
                    counts[i, self.kmer_to_int[kmer]] += 1
        return counts

    @property
    def dim(self) -> int:
        """Dimension of the k-mer count vector."""
        return len(self.kmers)

    @property
    def name(self) -> str:
        return f"{self.k}-mer Counts"


class CombinedExtractor(FeatureExtractor):
    """Combines features from multiple extractors.

    Args:
        extractors: A list of FeatureExtractor instances.
        device: "cpu" or "cuda". This will be passed to the extractors.
    """

    def __init__(self, extractors: list[FeatureExtractor], device: str = "cpu"):
        super().__init__()
        self.extractors = []
        self.device = device

        for ext in extractors:
            init_args = ext.__dict__
            cls = ext.__class__

            sig = inspect.signature(cls.__init__)
            valid_args = {k: v for k, v in init_args.items() if k in sig.parameters}
            valid_args["device"] = device

            self.extractors.append(cls(**valid_args))

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extract and concatenate features from all extractors.

        Args:
            sequences: List of RNA sequences.

        Returns:
            A 2D numpy or cupy array containing the concatenated features.
        """
        self._last_extracted_sequences = sequences
        all_features = [extractor.extract(sequences) for extractor in self.extractors]
        xp = _get_array_module(self.device)
        return xp.concatenate(all_features, axis=1)

    @property
    def dim(self) -> int:
        """Total dimension of the combined features."""
        return sum(extractor.dim for extractor in self.extractors)

    @property
    def name(self) -> str:
        """Name of the combined extractor."""
        return " + ".join(extractor.name for extractor in self.extractors)


class _SequenceDataset(Dataset):
    """Helper dataset to wrap a list of sequences for DataLoader."""

    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class _TokenizerCollate:
    """
    A collate function that initializes the Hugging Face tokenizer lazily
    within each worker process to avoid multiprocessing issues.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None

    def __call__(self, batch_sequences: list[str]):
        """Tokenize a batch of sequences."""
        if self.tokenizer is None:
            self.tokenizer = RnaTokenizer.from_pretrained(self.model_name)

        return self.tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )


class RNAFMExtractor(FeatureExtractor):
    """RNA-FM pretrained embedding extractor.

    Args:
        model_name: HuggingFace model name
        pooling: Pooling strategy ("mean", "cls", "max")
        device: Device to use ("cuda", "cpu", or "auto")
        batch_size: Batch size for extraction
        num_workers: Number of worker processes for data loading.
    """

    def __init__(
        self,
        model_name: str = "multimolecule/rnafm",
        pooling: Literal["mean", "cls", "max"] = "mean",
        device: str = "auto",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.num_workers = num_workers

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if self._device == "cpu":
            self.num_workers = 0

        self._model = None

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling to hidden states."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * mask
            hidden_states[mask == 0] = -1e9
            return hidden_states.max(dim=1)[0]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extract embeddings from RNA sequences."""
        self._last_extracted_sequences = sequences
        if self._model is None:
            self._model = RnaFmModel.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()

        dataset = _SequenceDataset(sequences)
        collate_obj = _TokenizerCollate(self.model_name)

        loader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_obj,
            num_workers=self.num_workers,
            pin_memory=self._device == "cuda",
        )

        all_embeddings = []
        with torch.no_grad():
            for batch_inputs in loader:
                inputs = {k: v.to(self._device) for k, v in batch_inputs.items()}
                outputs = self._model(**inputs)
                hidden_states = outputs.last_hidden_state
                pooled = self._pool(hidden_states, inputs["attention_mask"])
                all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def dim(self) -> int:
        """RNA-FM hidden dimension."""
        return 640

    @property
    def name(self) -> str:
        return f"RNA-FM ({self.pooling} pooling)"


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.conv2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1
        )
        self.conv3 = GATConv(
            hidden_channels * heads, out_channels, heads=1, dropout=0.1
        )
        self.lin = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x


class BPPMGATExtractor(FeatureExtractor):
    """BPPM-based GAT embedding extractor.

    Uses ViennaRNA for BPPM calculation and Graph Attention Network
    for structure-aware embeddings.
    """

    def __init__(
        self,
        out_channels: int = 64,
        hidden_channels: int = 32,
        heads: int = 4,
        device: str = "auto",
        max_length: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if self._device == "cpu":
            self.num_workers = 0

        self.model = GATModel(
            in_channels=4,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
        ).to(self._device)
        self.model.eval()

        self.alphabet = "ACGU"
        self.char_to_int = {char: i for i, char in enumerate(self.alphabet)}

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extract structure-aware embeddings.

        Args:
            sequences: List of RNA sequences

        Returns:
            Embedding matrix (N, out_channels)
        """
        self._last_extracted_sequences = sequences

        data_list = []
        for seq in sequences:
            seq = seq[: self.max_length]
            seq_len = len(seq)

            node_feats = torch.zeros(seq_len, 4)
            for i, char in enumerate(seq):
                if char in self.char_to_int:
                    node_feats[i, self.char_to_int[char]] = 1.0

            fc = RNA.fold_compound(seq)
            _ = fc.pf()
            bppm = fc.bpp()
            bppm = np.asarray(bppm)[1:, 1:]

            threshold = 0.01
            rows, cols = np.where(bppm > threshold)
            edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
            edge_attr = torch.tensor(bppm[rows, cols], dtype=torch.float)

            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        loader = PyGDataLoader(
            data_list,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self._device == "cuda",
        )
        all_embeddings_tensors = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                embeddings = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                all_embeddings_tensors.append(embeddings)

        final_tensor = torch.cat(all_embeddings_tensors, dim=0)

        if self._device == "cuda":
            xp = _get_array_module("cuda")
            return xp.asarray(final_tensor)
        else:
            return final_tensor.numpy()

    @property
    def dim(self) -> int:
        return self.out_channels

    @property
    def name(self) -> str:
        return "BPPM-GAT"
