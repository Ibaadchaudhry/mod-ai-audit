import re
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\b(i\s*(will|'ll|am\s+going\s+to|gonna)\s+(kill|murder|shoot|stab|hurt)\s+you)\b", re.IGNORECASE),
        re.compile(r"\b(you\s*(are|'re)\s+going\s+to\s+die)\b", re.IGNORECASE),
        re.compile(r"\b(i\s*(will|'ll|am\s+gonna|am\s+going\s+to)\s+find\s+where\s+you\s+live)\b", re.IGNORECASE),
        re.compile(r"\b(someone\s+should\s+(kill|shoot|stab|hurt)\s+you)\b", re.IGNORECASE),
        re.compile(r"\b(i\s*hope\s+you\s+die)\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\b(you\s+should\s+kill\s+yourself)\b", re.IGNORECASE),
        re.compile(r"\b(go\s+kill\s+yourself)\b", re.IGNORECASE),
        re.compile(r"\b(nobody\s+would\s+miss\s+you\s+if\s+you\s+died)\b", re.IGNORECASE),
        re.compile(r"\b(do\s+everyone\s+a\s+favo[u]?r\s+and\s+disappear)\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\b(i\s+(know|found)\s+where\s+you\s+live)\b", re.IGNORECASE),
        re.compile(r"\b(i\s*(will|'ll|am\s+going\s+to|gonna)\s+post\s+your\s+address)\b", re.IGNORECASE),
        re.compile(r"\b(i\s*(found|have)\s+your\s+real\s+name)\b", re.IGNORECASE),
        re.compile(r"\b(everyone\s+will\s+know\s+who\s+you\s+really\s+are)\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b\w+\s+are\s+not\s+(?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+animals\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+should\s+be\s+exterminated\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+a\s+disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\b(?=everyone\s+report)everyone\s+report\s+@?\w+\b", re.IGNORECASE),
        re.compile(r"\b(let'?s\s+all\s+go\s+after\s+@?\w+)\b", re.IGNORECASE),
        re.compile(r"\b(mass\s+report\s+this\s+account)\b", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> Optional[Dict]:
    candidate = "" if text is None else str(text)
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(candidate):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


class _TransformerScoreEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, model_dir: str, max_length: int = 128, batch_size: int = 64):
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.to(self._device)
            self._model.eval()

    # Explicit sklearn compatibility helpers for strict clone checks.
    def get_params(self, deep=True):
        return {
            "model_dir": self.model_dir,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reset lazy-loaded artifacts when params change.
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def fit(self, X, y):
        # No trainable params for the wrapped transformer in this adapter.
        self._ensure_loaded()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1
        return self

    def predict_proba(self, X):
        self._ensure_loaded()
        texts = ["" if t is None else str(t) for t in list(X)]
        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.append(probs)

        p1 = np.concatenate(all_probs)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def _more_tags(self):
        return {
            "binary_only": True,
            "requires_fit": True,
        }


class ModerationPipeline:
    def __init__(
        self,
        model_dir: str,
        allow_threshold: float = 0.4,
        block_threshold: float = 0.6,
        max_length: int = 128,
        batch_size: int = 64,
    ):
        self.model_dir = model_dir
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.calibrator = None

    def _raw_model_probs(self, texts: List[str]) -> np.ndarray:
        normalized = ["" if t is None else str(t) for t in texts]
        all_probs: List[np.ndarray] = []
        for i in range(0, len(normalized), self.batch_size):
            batch = normalized[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def fit_calibrator(self, texts: List[str], labels: List[int], cv: int = 3):
        base_estimator = _TransformerScoreEstimator(
            model_dir=self.model_dir,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method="isotonic",
            cv=cv,
        )
        self.calibrator.fit(list(texts), np.asarray(labels).astype(int))
        return self

    def _calibrated_confidence(self, text: str) -> float:
        if self.calibrator is None:
            return float(self._raw_model_probs([text])[0])
        proba = self.calibrator.predict_proba([text])[0, 1]
        return float(proba)

    def predict(self, text: str) -> Dict:
        decision = input_filter(text)
        if decision is not None:
            return decision

        confidence = self._calibrated_confidence(text)
        if confidence >= self.block_threshold:
            return {"decision": "block", "layer": "model", "confidence": confidence}
        if confidence <= self.allow_threshold:
            return {"decision": "allow", "layer": "model", "confidence": confidence}
        return {"decision": "review", "layer": "model", "confidence": confidence}
