from .activitynet_qa import ActivitynetQADataset
from .base import BaseEvalDataset
from .charades_sta import CharadesSTADataset
from .egoschema import EgoSchemaDataset
from .longvideobench import LongVideoBenchDataset
from .lvbench import LVBenchDataset
from .mlvu import MLVUDataset
from .mmvu import MMVUDataset
from .mvbench import MVBenchDataset
from .nextqa import NextQADataset
from .perception_test import PerceptionTestDataset
from .tempcompass import TempCompassDataset
from .videomme import VideoMMEDataset


DATASET_REGISTRY = {
    "videomme": VideoMMEDataset,
    "mmvu": MMVUDataset,
    "mvbench": MVBenchDataset,
    "egoschema": EgoSchemaDataset,
    "perception_test": PerceptionTestDataset,
    "activitynet_qa": ActivitynetQADataset,
    "mlvu": MLVUDataset,
    "longvideobench": LongVideoBenchDataset,
    "lvbench": LVBenchDataset,
    "tempcompass": TempCompassDataset,
    "nextqa": NextQADataset,
    "charades_sta": CharadesSTADataset,
}


def build_dataset(benchmark_name: str, **kwargs) -> BaseEvalDataset:
    assert benchmark_name in DATASET_REGISTRY, f"Unknown benchmark: {benchmark_name}, available: {DATASET_REGISTRY.keys()}"
    return DATASET_REGISTRY[benchmark_name](**kwargs)
