# Copyright GrokCV. All rights reserved.
import numpy as np
import xml.etree.ElementTree as ET
from typing import List

from deepir.datasets.sirst_voc_det import SIRSTVOCDetDataset
from deepir.registry import DATASETS


@DATASETS.register_module()
class SIRSTVOCDetSafeDataset(SIRSTVOCDetDataset):
    """SIRST VOC dataset variant that safely ignores placeholder empty boxes.

    This is needed for open-sirst-v2, whose negative images may contain an
    ``object`` tag without any ``bndbox``.
    """

    def _parse_instance_info(
        self, raw_ann_info: ET, minus_one: bool = True
    ) -> List[dict]:
        instances = []
        for obj in raw_ann_info.findall("object"):
            name = obj.find("name").text
            if name not in self._metainfo["classes"]:
                continue

            difficult = obj.find("difficult")
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find("bndbox")
            if bnd_box is None:
                continue

            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
            if minus_one:
                bbox = [x - 1 for x in bbox]

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.bbox_min_size or h < self.bbox_min_size:
                    ignore = True

            instance = {}
            instance["ignore_flag"] = 1 if (difficult or ignore) else 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[name]
            instances.append(instance)

        if not instances:
            return []

        return instances
