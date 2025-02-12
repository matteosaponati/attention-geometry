
import gzip
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    label: bool

    def json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
        }

    @staticmethod
    def from_json(d: dict) -> 'Sample':
        return Sample(**d)


@dataclass(frozen=True)
class ScoredSample:
    id: str
    score: float

    def json(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
        }

    @staticmethod
    def from_json(d: dict) -> 'ScoredSample':
        return ScoredSample(**d)


@dataclass(frozen=True)
class TrainDataset:
    name: str
    train_samples: List[Sample]
    dev_samples: Optional[List[Sample]] = None

    def json(self) -> dict:
        return {
            "name": self.name,
            "train_samples": [s.json() for s in self.train_samples],
            "dev_samples": [s.json() for s in self.dev_samples] if self.dev_samples is not None else None,
        }

    @staticmethod
    def from_json(d: dict) -> 'TrainDataset':
        return TrainDataset(
            name=d['name'],
            train_samples=[Sample.from_json(s) for s in d['train_samples']],
            dev_samples=[Sample.from_json(s) for s in d['dev_samples']] if d['dev_samples'] is not None else None,
        )

    def save(self, path: Path) -> None:
        with gzip.open(path, 'wt') as fout:
            fout.write(json.dumps(obj=self.json(), indent=2))
            fout.write('\n')

    @staticmethod
    def load(path: Path) -> 'TrainDataset':
        with gzip.open(path, 'rt') as fin:
            data = json.load(fin)
        return TrainDataset.from_json(data)


@dataclass(frozen=True)
class TestDataset:
    name: str
    test_samples: List[Sample]

    def json(self) -> dict:
        return {
            "name": self.name,
            "test_samples": [s.json() for s in self.test_samples],
        }

    @staticmethod
    def from_json(d: dict) -> 'TestDataset':
        return TestDataset(
            name=d['name'],
            test_samples=[Sample.from_json(s) for s in d['test_samples']],
        )

    def save(self, path: Path) -> None:
        with gzip.open(path, 'wt') as fout:
            fout.write(json.dumps(obj=self.json(), indent=2))
            fout.write('\n')

    @staticmethod
    def load(path: Path) -> 'TestDataset':
        with gzip.open(path, 'rt') as fin:
            data = json.load(fin)
        return TestDataset.from_json(data)


@dataclass(frozen=True)
class ScoredDataset:
    classifier_name: str
    train_data: str
    test_data: str
    classifier_params: dict
    default_threshold: float
    scores: List[ScoredSample]

    def json(self) -> dict:
        return {
            "classifier_name": self.classifier_name,
            "train_data": self.train_data,
            "test_data": self.test_data,
            "classifier_params": self.classifier_params,
            "default_threshold": self.default_threshold,
            "scores": [s.json() for s in self.scores],
        }

    @staticmethod
    def from_json(d: dict) -> 'ScoredDataset':
        d['scores'] = [ScoredSample.from_json(s) for s in d['scores']]
        return ScoredDataset(**d)

    def save(self, path: Path) -> None:
        with gzip.open(path, 'wt') as fout:
            fout.write(json.dumps(obj=self.json(), indent=2))
            fout.write('\n')

    @staticmethod
    def load(path: Path) -> 'ScoredDataset':
        with gzip.open(path, 'rt') as fin:
            data = json.load(fin)
        return ScoredDataset.from_json(data)