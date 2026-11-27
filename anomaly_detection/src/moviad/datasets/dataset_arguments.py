from dataclasses import dataclass
from typing import Optional, Union
from moviad.utilities.configurations import Split

@dataclass
class DatasetArguments:
    dataset_path: str
    img_size: Optional[Union[tuple, list]] = None
    gt_mask_size: Optional[Union[tuple, list]] = None
    image_transform_list: Optional[list] = None
    # test set folder (labels.csv, normal/, anomalous/, masks/), if the
    # dataset supports it (VarroaDataset); None = dataset default
    test_path: Optional[str] = None
    # True = use all normal images of the test set, without the
    # _MAX_TEST_NORMALS subset of VarroaDataset
    full_test: bool = False
