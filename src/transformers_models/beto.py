"""Fine-tuning de BETO para Aspect-Based Sentiment Analysis.

Implementa el formato de auxiliary sentence (Sun et al., 2019):
    [CLS] reseña [SEP] aspecto [SEP]
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "dccuchile/bert-base-spanish-wwm-uncased"
DEFAULT_MAX_LENGTH: int = 128
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 3
DEFAULT_LEARNING_RATE: float = 2e-5
DEFAULT_WEIGHT_DECAY: float = 0.01

LABEL2ID: dict[str, int] = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class AspectDataset:
    """Dataset PyTorch para pares (reseña, aspecto) -> label."""

    def __init__(
        self,
        texts: list[str],
        aspects: list[str],
        labels: list[str] | None,
        tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Inicializa el dataset.

        Args:
            texts: Lista de textos.
            aspects: Lista de aspectos paralelos.
            labels: Etiquetas en {'pos','neg','neu'} o None para inferencia.
            tokenizer: Tokenizer de Hugging Face.
            max_length: Longitud máxima de tokens.
        """
        if len(texts) != len(aspects):
            raise ValueError("texts y aspects deben tener la misma longitud")
        if labels is not None and len(labels) != len(texts):
            raise ValueError("texts, aspects y labels deben tener la misma longitud")
        if labels is not None:
            invalid = sorted({lbl for lbl in labels if lbl not in LABEL2ID})
            if invalid:
                raise ValueError(
                    f"Etiquetas no reconocidas: {invalid}. Esperadas: {sorted(LABEL2ID)}"
                )
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        import torch

        encoding = self.tokenizer(
            self.texts[idx],
            self.aspects[idx],
            truncation="only_first",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(LABEL2ID[self.labels[idx]], dtype=torch.long)
        return item


class BETOAspectClassifier:
    """Clasificador BETO fine-tuneado para ABSA en español."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 3,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        """Inicializa modelo y tokenizer.

        Args:
            model_name: Identificador HF del modelo.
            num_labels: Número de clases.
            max_length: Tokens máximos.
            device: 'cuda', 'mps' o 'cpu'; None auto-detecta.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("Usando dispositivo: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

    def fit(
        self,
        texts: list[str],
        aspects: list[str],
        labels: list[str],
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Entrena BETO sobre los datos proporcionados.

        Args:
            texts: Textos de entrenamiento.
            aspects: Aspectos paralelos.
            labels: Labels en {'pos','neg','neu'}.
            epochs: Número de épocas.
            batch_size: Tamaño de batch.
            learning_rate: Tasa de aprendizaje.
            weight_decay: Decay para AdamW.
            seed: Semilla opcional para inicialización y orden de batches.

        Returns:
            Diccionario con historial de pérdida por época.
        """
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader

        generator = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            generator = torch.Generator()
            generator.manual_seed(seed)

        dataset = AspectDataset(texts, aspects, labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history: dict[str, list[float]] = {"loss": []}
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(loader))
            history["loss"].append(avg)
            logger.info("Época %d/%d - loss=%.4f", epoch + 1, epochs, avg)
        return history

    def predict(self, texts: list[str], aspects: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[str]:
        """Predice etiquetas para pares (texto, aspecto).

        Args:
            texts: Textos.
            aspects: Aspectos.
            batch_size: Tamaño de batch.

        Returns:
            Lista de etiquetas predichas.
        """
        import torch
        from torch.utils.data import DataLoader

        if not texts:
            return []

        dataset = AspectDataset(texts, aspects, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        return [ID2LABEL[int(p)] for p in preds]

    def predict_proba(self, texts: list[str], aspects: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Devuelve probabilidades softmax.

        Args:
            texts: Textos.
            aspects: Aspectos.
            batch_size: Tamaño de batch.

        Returns:
            Array (N, num_labels).
        """
        import torch
        from torch.utils.data import DataLoader

        if not texts:
            return np.empty((0, self.num_labels))

        dataset = AspectDataset(texts, aspects, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        probas: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probas.append(probs)
        return np.vstack(probas)

    def save(self, path: str | Path) -> None:
        """Guarda modelo, tokenizer y configuración del clasificador.

        Args:
            path: Directorio de salida.
        """
        import json as _json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with (path / "classifier_config.json").open("w", encoding="utf-8") as fh:
            _json.dump({"max_length": self.max_length}, fh)
        logger.info("BETO guardado en %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "BETOAspectClassifier":
        """Carga un modelo BETO guardado.

        Args:
            path: Directorio del modelo.
            device: Dispositivo a usar.

        Returns:
            Instancia con modelo cargado.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import json as _json
        import torch

        path = Path(path)
        instance = cls.__new__(cls)
        instance.model_name = str(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.num_labels = instance.model.config.num_labels

        config_file = path / "classifier_config.json"
        if config_file.exists():
            with config_file.open("r", encoding="utf-8") as fh:
                cfg = _json.load(fh)
            instance.max_length = int(cfg.get("max_length", DEFAULT_MAX_LENGTH))
        else:
            instance.max_length = DEFAULT_MAX_LENGTH

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        instance.device = torch.device(device)
        instance.model.to(instance.device)
        logger.info("BETO cargado desde %s", path)
        return instance
