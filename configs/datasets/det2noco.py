import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List

from PIL import Image
import numpy as np

import mmcv
from mmengine.logging import print_log
from mmengine.fileio import list_from_file, get_local_path, get
from mmengine.registry import DATASETS
from mmdet.datasets.xml_style import XMLDataset
from deepir.evaluation.metrics.mnocoap_det_metric import det_results_to_noco_centroids
from deepir.evaluation.metrics.mean_nocoap import eval_mnocoap
from deepir.evaluation.metrics.Seg2DetTargets import NoCoTargets

@DATASETS.register_module()
class SIRSTDet2NoCoDataset(XMLDataset):
    """SIRST dataset for bbox detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: mixed.
        ann_subdir (str): Subdir where annotations are. Default: annotations/bboxes.
    """
    METAINFO = {
        'classes': ('Target',),
    }

    def __init__(self,
                 min_size=None,
                 img_subdir='mixed',
                 ann_subdir='annotations/bboxes',
                 gt_noco_map_loader_cfg=None,
                 noco_thrs=None,
                 noco_mode='det2noco',
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super(SIRSTDet2NoCoDataset, self).__init__(
            img_subdir=img_subdir,
            ann_subdir=ann_subdir,
            **kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo['classes'])}
        self.min_size = min_size
        # NoCoTargets for generating ground truth noco maps
        self.noco_targets = NoCoTargets()
        if noco_thrs is None:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            # noco_thrs = np.array([0.5])
            noco_thrs = [noco_thrs] if isinstance(
                noco_thrs, float) else noco_thrs
        self.noco_thrs = noco_thrs
        assert noco_mode in ['det2noco', 'noco_peak']
        self.noco_mode = noco_mode
        self.best_mnocoap = -np.inf

        self.gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
                            for i in range(len(self))]
        self.gt_bboxes = [self.get_ann_info(i)['bboxes']
                          for i in range(len(self))]

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            # img_subdir='PNGImages'
            file_name = osp.join(self.img_subdir, f'{img_id}.png')
            # Use data_root to construct absolute path
            xml_path = osp.join(self.data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    def parse_data_info(self, img_info: dict):
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information.

        Returns:
            dict: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.data_root, img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes)
            height, width = img.shape[:2]

        data_info['height'] = height
        data_info['width'] = width
        data_info['instances'] = self._parse_xml_data(root)
        return data_info

    def _parse_xml_data(self, root):
        """Parse xml annotation file.

        Args:
            root (Element): The root element of xml file.

        Returns:
            list[dict]: Annotation info.
        """
        instances = []
        for obj in root.findall('object'):
            instance = {}
            name = obj.find('name').text
            if name not in self._metainfo['classes']:
                continue
            instance['bbox_label'] = self.cat2label[name]
            instance['ignore_flag'] = False
            bnd_box = obj.find('bndbox')
            if bnd_box is not None:
                # XML uses 1-based indexing, convert to 0-based
                instance['bbox'] = [
                    int(float(bnd_box.find('xmin').text)) - 1,
                    int(float(bnd_box.find('ymin').text)) - 1,
                    int(float(bnd_box.find('xmax').text)) - 1,
                    int(float(bnd_box.find('ymax').text)) - 1
                ]
                difficult = obj.find('difficult')
                if difficult is not None:
                    instance['ignore_flag'] = int(difficult.text) == 1
                instances.append(instance)
        return instances

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join(self.img_subdir, f'{img_id}.png')
            xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=0):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self._metainfo['classes']:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from data_list by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        data_info = self.get_data_info(idx)
        instances = data_info.get('instances', [])

        bboxes = []
        labels = []
        for instance in instances:
            if instance.get('ignore_flag', False):
                continue
            bboxes.append(instance['bbox'])
            labels.append(instance['bbox_label'])

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        ann = dict(bboxes=bboxes, labels=labels)
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self._metainfo['classes']:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def get_gt_noco_map_by_idx(self, index):
        """Get one ground truth normalized contrast map for evaluation."""
        data_info = self.get_data_info(index)
        img_path = data_info['img_path']

        # Load image
        from mmengine.fileio import get
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, backend='cv2')

        # Get bboxes
        ann_info = self.get_ann_info(index)
        # Convert bboxes to list format
        gt_bboxes = []
        if len(ann_info['bboxes']) > 0:
            for bbox in ann_info['bboxes']:
                gt_bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        # Generate noco map using NoCoTargets
        gt_noco_map = self.noco_targets.generate_gt_noco_map(img, gt_bboxes)
        return gt_noco_map

    def evaluate(self,
                 results,
                 metric='mNoCoAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in SIRST's NoCo protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mNoCoAP'.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: NoCoAP/AP/recall metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mNoCoAP',]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        if metric == ['mNoCoAP']:
            # if noco_thrs is None:
            #     noco_thrs = np.linspace(
            #         .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            #     noco_thrs = [noco_thrs] if isinstance(
            #         noco_thrs, float) else noco_thrs

            # prepare inputs for eval_mnocoap
            # if self.noco_mode == 'det2noco':
            det_centroids = det_results_to_noco_centroids(results)
            # else:
            #     det_centroids = det_results_to_noco_peaks(results)
            # gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
            #                 for i in range(len(self))]
            # gt_bboxes = [self.get_ann_info(i)['bboxes']
            #              for i in range(len(self))]

            # compute mNoCoAP
            eval_results = OrderedDict()
            mean_nocoaps = []
            for noco_thr in self.noco_thrs:
                print_log(f'\n{"-" * 15}noco_thr: {noco_thr}{"-" * 15}')
                mean_nocoap, _ = eval_mnocoap(
                    det_centroids, self.gt_noco_maps, self.gt_bboxes,
                    noco_thr=noco_thr, logger=logger)
                # mean_nocoap, _ = eval_mnocoap(
                #     det_centroids, gt_noco_maps, gt_bboxes,
                #     noco_thr=noco_thr, logger=logger)
                mean_nocoaps.append(mean_nocoap)
                eval_results[f'NoCoAP{int(noco_thr * 100):02d}'] = round(
                    mean_nocoap, 3)
            eval_results['mNoCoAP'] = sum(mean_nocoaps) / len(mean_nocoaps)
            print("current eval_results['mNoCoAP']:", eval_results['mNoCoAP'])
            if self.best_mnocoap < eval_results['mNoCoAP']:
                self.best_mnocoap = eval_results['mNoCoAP']
            print("best eval_results['mNoCoAP']:", self.best_mnocoap)
            print_log(f"\n best eval_results['mNoCoAP']: {self.best_mnocoap}")
        else:
            raise ValueError(
                f"unsupported metric {metric}")
        return eval_results