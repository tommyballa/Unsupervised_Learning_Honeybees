from dataclasses import dataclass
from typing import Optional, Union
from moviad.utilities.configurations import Split

@dataclass
class DatasetArguments:
    dataset_path: str
    img_size: Optional[Union[tuple, list]] = None
    gt_mask_size: Optional[Union[tuple, list]] = None
    image_transform_list: Optional[list] = None
    # cartella del test set (labels.csv, normal/, anomalous/, masks/), se il
    # dataset la supporta (VarroaDataset); None = default del dataset
    test_path: Optional[str] = None
    # True = usa tutte le normali del test set, senza il sottoinsieme di
    # _MAX_TEST_NORMALS di VarroaDataset
    full_test: bool = False
