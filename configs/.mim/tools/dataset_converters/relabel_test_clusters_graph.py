import argparse
import copy
import json
import os
import os.path as osp
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class BoxComponent:
    indices: List[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Relabel test-set cluster annotations into a separate output directory.')
    parser.add_argument(
        '--split-file',
        default='/opt/data/private/cjt/data/DenseSIRST/SIRSTdevkit/Splits/test_v2.txt',
        help='Image id list for the test split.',
    )
    parser.add_argument(
        '--bbox-dir',
        default='/opt/data/private/cjt/data/DenseSIRST/SIRSTdevkit/test/SIRST/BBOX',
        help='Directory of instance-level VOC XML annotations.',
    )
    parser.add_argument(
        '--out-dir',
        default='/opt/data/private/cjt/data/DenseSIRST/SIRSTdevkit/test/SIRST/Cluster_relabel_graph_v1',
        help='Directory to save relabeled cluster XML files.',
    )
    parser.add_argument(
        '--min-component-size',
        type=int,
        default=2,
        help='Minimum number of objects required to form a cluster.',
    )
    parser.add_argument(
        '--base-expand',
        type=float,
        default=8.0,
        help='Base box expansion in pixels for graph construction.',
    )
    parser.add_argument(
        '--size-expand-ratio',
        type=float,
        default=2.0,
        help='Per-box expansion ratio relative to the box width/height.',
    )
    parser.add_argument(
        '--context-pad',
        type=float,
        default=2.0,
        help='Minimum padding added around the final cluster envelope.',
    )
    parser.add_argument(
        '--context-ratio',
        type=float,
        default=0.12,
        help='Padding ratio relative to the component envelope size.',
    )
    parser.add_argument(
        '--min-split-gap',
        type=float,
        default=18.0,
        help='Absolute gap threshold for recursively splitting chained components.',
    )
    parser.add_argument(
        '--split-gap-ratio',
        type=float,
        default=4.0,
        help='Gap ratio relative to median object size for recursive splitting.',
    )
    parser.add_argument(
        '--min-split-size',
        type=int,
        default=4,
        help='Only components with at least this many objects can be recursively split.',
    )
    parser.add_argument(
        '--summary-json',
        default='summary.json',
        help='Summary filename saved inside out-dir.',
    )
    return parser.parse_args()


def load_split_ids(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def list_ids_from_bbox_dir(bbox_dir: str) -> List[str]:
    names = []
    for file_name in sorted(os.listdir(bbox_dir)):
        if file_name.endswith('.xml'):
            names.append(osp.splitext(file_name)[0])
    return names


def resolve_img_ids(split_file: str, bbox_dir: str) -> List[str]:
    available = set(list_ids_from_bbox_dir(bbox_dir))
    if not split_file or not osp.isfile(split_file):
        return sorted(available)

    split_ids = load_split_ids(split_file)
    filtered = [img_id for img_id in split_ids if img_id in available]
    if filtered:
        return filtered
    return sorted(available)


def parse_voc_annotation(xml_path: str) -> Tuple[ET.ElementTree, ET.Element, str, Dict[str, int], np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext('filename', default=osp.splitext(osp.basename(xml_path))[0]).strip()
    size_node = root.find('size')
    if size_node is None:
        raise ValueError(f'Missing <size> in {xml_path}')

    img_w = int(float(size_node.findtext('width', default='0')))
    img_h = int(float(size_node.findtext('height', default='0')))
    img_d = int(float(size_node.findtext('depth', default='1')))
    size = dict(width=img_w, height=img_h, depth=img_d)

    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        coords = []
        valid = True
        for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
            node = bndbox.find(tag)
            if node is None or node.text is None:
                valid = False
                break
            coords.append(float(node.text))
        if not valid:
            continue
        boxes.append(coords)

    if len(boxes) == 0:
        box_array = np.zeros((0, 4), dtype=np.float32)
    else:
        box_array = np.asarray(boxes, dtype=np.float32)

    return tree, root, filename, size, box_array


def box_centers(boxes: np.ndarray) -> np.ndarray:
    centers = np.zeros((boxes.shape[0], 2), dtype=np.float32)
    centers[:, 0] = 0.5 * (boxes[:, 0] + boxes[:, 2])
    centers[:, 1] = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return centers


def build_expanded_boxes(
    boxes: np.ndarray,
    img_w: int,
    img_h: int,
    base_expand: float,
    size_expand_ratio: float,
) -> np.ndarray:
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1.0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1.0, a_max=None)
    expand_x = np.maximum(base_expand, widths * size_expand_ratio)
    expand_y = np.maximum(base_expand, heights * size_expand_ratio)

    expanded = boxes.copy()
    expanded[:, 0] -= expand_x
    expanded[:, 1] -= expand_y
    expanded[:, 2] += expand_x
    expanded[:, 3] += expand_y
    expanded[:, 0::2] = np.clip(expanded[:, 0::2], 0.0, float(img_w))
    expanded[:, 1::2] = np.clip(expanded[:, 1::2], 0.0, float(img_h))
    return expanded


def boxes_intersect(box1: np.ndarray, box2: np.ndarray) -> bool:
    return not (
        box1[2] < box2[0] or
        box2[2] < box1[0] or
        box1[3] < box2[1] or
        box2[3] < box1[1]
    )


def connected_components(expanded_boxes: np.ndarray) -> List[BoxComponent]:
    num_boxes = int(expanded_boxes.shape[0])
    if num_boxes == 0:
        return []

    visited = np.zeros((num_boxes,), dtype=np.uint8)
    components: List[BoxComponent] = []

    for start in range(num_boxes):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = 1
        indices = []

        while queue:
            current = queue.pop()
            indices.append(current)
            current_box = expanded_boxes[current]
            for other in range(num_boxes):
                if visited[other]:
                    continue
                if boxes_intersect(current_box, expanded_boxes[other]):
                    visited[other] = 1
                    queue.append(other)

        components.append(BoxComponent(indices=sorted(indices)))

    return components


def recursive_split_component(
    boxes: np.ndarray,
    component: Sequence[int],
    min_split_gap: float,
    split_gap_ratio: float,
    min_split_size: int,
) -> List[List[int]]:
    indices = list(component)
    if len(indices) < min_split_size:
        return [indices]

    sub_boxes = boxes[indices]
    span_x = float(sub_boxes[:, 2].max() - sub_boxes[:, 0].min())
    span_y = float(sub_boxes[:, 3].max() - sub_boxes[:, 1].min())
    axis = 0 if span_x >= span_y else 1
    start_col = axis
    end_col = axis + 2

    centers = 0.5 * (sub_boxes[:, start_col] + sub_boxes[:, end_col])
    order_local = np.argsort(centers)
    ordered_indices = [indices[i] for i in order_local]
    ordered_boxes = boxes[ordered_indices]

    gaps = ordered_boxes[1:, start_col] - ordered_boxes[:-1, end_col]
    if gaps.size == 0:
        return [indices]

    box_sizes = ordered_boxes[:, end_col] - ordered_boxes[:, start_col]
    median_size = float(np.median(np.clip(box_sizes, a_min=1.0, a_max=None)))
    gap_threshold = max(float(min_split_gap), float(split_gap_ratio) * median_size)
    split_pos = int(np.argmax(gaps))
    max_gap = float(gaps[split_pos])

    if max_gap <= gap_threshold:
        return [indices]

    left = ordered_indices[:split_pos + 1]
    right = ordered_indices[split_pos + 1:]
    if len(left) < 2 or len(right) < 2:
        return [indices]

    left_parts = recursive_split_component(
        boxes,
        left,
        min_split_gap=min_split_gap,
        split_gap_ratio=split_gap_ratio,
        min_split_size=min_split_size,
    )
    right_parts = recursive_split_component(
        boxes,
        right,
        min_split_gap=min_split_gap,
        split_gap_ratio=split_gap_ratio,
        min_split_size=min_split_size,
    )
    return left_parts + right_parts


def component_to_cluster_box(
    boxes: np.ndarray,
    component: Sequence[int],
    img_w: int,
    img_h: int,
    context_pad: float,
    context_ratio: float,
) -> np.ndarray:
    sub_boxes = boxes[list(component)]
    x1 = float(sub_boxes[:, 0].min())
    y1 = float(sub_boxes[:, 1].min())
    x2 = float(sub_boxes[:, 2].max())
    y2 = float(sub_boxes[:, 3].max())

    span_w = max(1.0, x2 - x1)
    span_h = max(1.0, y2 - y1)
    widths = np.clip(sub_boxes[:, 2] - sub_boxes[:, 0], a_min=1.0, a_max=None)
    heights = np.clip(sub_boxes[:, 3] - sub_boxes[:, 1], a_min=1.0, a_max=None)

    pad_x = max(float(context_pad), float(context_ratio) * span_w, float(np.median(widths)))
    pad_y = max(float(context_pad), float(context_ratio) * span_h, float(np.median(heights)))

    cluster = np.array([
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(img_w), x2 + pad_x),
        min(float(img_h), y2 + pad_y),
    ], dtype=np.float32)
    return cluster


def deduplicate_boxes(boxes: List[np.ndarray]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    merged: List[np.ndarray] = []
    for box in boxes:
        keep = True
        for existing in merged:
            if np.allclose(box, existing, atol=1.0):
                keep = False
                break
        if keep:
            merged.append(box)

    result = np.stack(merged, axis=0).astype(np.float32)
    order = np.lexsort((result[:, 0], result[:, 1]))
    return result[order]


def generate_cluster_boxes(
    boxes: np.ndarray,
    img_w: int,
    img_h: int,
    min_component_size: int,
    base_expand: float,
    size_expand_ratio: float,
    context_pad: float,
    context_ratio: float,
    min_split_gap: float,
    split_gap_ratio: float,
    min_split_size: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), dict(num_objects=0, num_components=0, num_clusters=0)

    boxes = boxes.copy().astype(np.float32)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(img_w))
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(img_h))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]

    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), dict(num_objects=0, num_components=0, num_clusters=0)

    expanded_boxes = build_expanded_boxes(
        boxes,
        img_w=img_w,
        img_h=img_h,
        base_expand=base_expand,
        size_expand_ratio=size_expand_ratio,
    )
    components = connected_components(expanded_boxes)

    cluster_boxes: List[np.ndarray] = []
    for component in components:
        split_parts = recursive_split_component(
            boxes,
            component.indices,
            min_split_gap=min_split_gap,
            split_gap_ratio=split_gap_ratio,
            min_split_size=min_split_size,
        )
        for part in split_parts:
            if len(part) < min_component_size:
                continue
            cluster_box = component_to_cluster_box(
                boxes,
                part,
                img_w=img_w,
                img_h=img_h,
                context_pad=context_pad,
                context_ratio=context_ratio,
            )
            if cluster_box[2] > cluster_box[0] and cluster_box[3] > cluster_box[1]:
                cluster_boxes.append(cluster_box)

    cluster_array = deduplicate_boxes(cluster_boxes)
    summary = dict(
        num_objects=int(boxes.shape[0]),
        num_components=int(len(components)),
        num_clusters=int(cluster_array.shape[0]),
    )
    return cluster_array, summary


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = '\n' + level * '  '
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + '  '
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def build_output_tree(
    src_root: ET.Element,
    cluster_boxes: np.ndarray,
) -> ET.ElementTree:
    new_root = ET.Element('annotation')

    for child in list(src_root):
        if child.tag == 'cluster':
            continue
        new_root.append(copy.deepcopy(child))

    for box in cluster_boxes:
        cluster_node = ET.SubElement(new_root, 'cluster')
        name_node = ET.SubElement(cluster_node, 'name')
        name_node.text = 'Target'

        bndbox = ET.SubElement(cluster_node, 'bndbox')
        x1, y1, x2, y2 = np.round(box).astype(np.int32)
        for tag, value in (
            ('xmin', x1),
            ('ymin', y1),
            ('xmax', x2),
            ('ymax', y2),
        ):
            coord = ET.SubElement(bndbox, tag)
            coord.text = str(int(value))

    indent_xml(new_root)
    return ET.ElementTree(new_root)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img_ids = resolve_img_ids(args.split_file, args.bbox_dir)
    records = []

    for img_id in img_ids:
        bbox_xml = osp.join(args.bbox_dir, f'{img_id}.xml')
        if not osp.isfile(bbox_xml):
            raise FileNotFoundError(f'Instance annotation not found: {bbox_xml}')

        _, src_root, _, size, boxes = parse_voc_annotation(bbox_xml)
        cluster_boxes, summary = generate_cluster_boxes(
            boxes,
            img_w=size['width'],
            img_h=size['height'],
            min_component_size=args.min_component_size,
            base_expand=args.base_expand,
            size_expand_ratio=args.size_expand_ratio,
            context_pad=args.context_pad,
            context_ratio=args.context_ratio,
            min_split_gap=args.min_split_gap,
            split_gap_ratio=args.split_gap_ratio,
            min_split_size=args.min_split_size,
        )

        out_tree = build_output_tree(src_root, cluster_boxes)
        out_path = osp.join(args.out_dir, f'{img_id}_with_clusters.xml')
        out_tree.write(out_path, encoding='utf-8', xml_declaration=False)

        cluster_area = float(((cluster_boxes[:, 2] - cluster_boxes[:, 0]) * (
            cluster_boxes[:, 3] - cluster_boxes[:, 1])).sum()) if cluster_boxes.size > 0 else 0.0
        object_centers = box_centers(boxes) if boxes.size > 0 else np.zeros((0, 2), dtype=np.float32)
        covered = 0
        for center in object_centers:
            if cluster_boxes.size == 0:
                continue
            inside = (
                (cluster_boxes[:, 0] <= center[0]) &
                (cluster_boxes[:, 1] <= center[1]) &
                (cluster_boxes[:, 2] >= center[0]) &
                (cluster_boxes[:, 3] >= center[1])
            )
            if bool(inside.any()):
                covered += 1

        records.append(dict(
            img_id=img_id,
            xml_path=out_path,
            num_objects=int(boxes.shape[0]),
            num_components=summary['num_components'],
            num_clusters=summary['num_clusters'],
            center_coverage=float(covered / max(int(boxes.shape[0]), 1)),
            cluster_area_ratio=float(cluster_area / max(size['width'] * size['height'], 1)),
        ))

    summary = dict(
        split_file=args.split_file,
        bbox_dir=args.bbox_dir,
        out_dir=args.out_dir,
        params=dict(
            min_component_size=args.min_component_size,
            base_expand=args.base_expand,
            size_expand_ratio=args.size_expand_ratio,
            context_pad=args.context_pad,
            context_ratio=args.context_ratio,
            min_split_gap=args.min_split_gap,
            split_gap_ratio=args.split_gap_ratio,
            min_split_size=args.min_split_size,
        ),
        num_images=len(records),
        avg_clusters=float(np.mean([record['num_clusters'] for record in records])) if records else 0.0,
        avg_center_coverage=float(np.mean([record['center_coverage'] for record in records])) if records else 0.0,
        avg_cluster_area_ratio=float(np.mean([record['cluster_area_ratio'] for record in records])) if records else 0.0,
        records=records,
    )

    summary_path = osp.join(args.out_dir, args.summary_json)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'num_images': summary['num_images'],
        'avg_clusters': round(summary['avg_clusters'], 4),
        'avg_center_coverage': round(summary['avg_center_coverage'], 4),
        'avg_cluster_area_ratio': round(summary['avg_cluster_area_ratio'], 6),
        'out_dir': args.out_dir,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
