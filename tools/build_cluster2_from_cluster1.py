import argparse
import json
import os
import os.path as osp
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build Cluster2 by supplementing Cluster1 with singleton clusters '
                    'for instances not covered by any existing cluster.')
    parser.add_argument(
        '--bbox-dir',
        default='data/DenseSIRST/SIRSTdevkit/SIRST/BBox',
        help='Directory with instance-level VOC XML annotations.')
    parser.add_argument(
        '--cluster1-dir',
        default='data/DenseSIRST/SIRSTdevkit/SIRST/Cluster1',
        help='Directory with the original Cluster1 XML annotations.')
    parser.add_argument(
        '--out-dir',
        default='data/DenseSIRST/SIRSTdevkit/SIRST/Cluster2',
        help='Directory to save the supplemented Cluster2 XML annotations.')
    parser.add_argument(
        '--cluster-suffix',
        default='_with_clusters.xml',
        help='Cluster XML suffix.')
    parser.add_argument(
        '--target-name',
        default='Target',
        help='Only objects/clusters with this name are considered.')
    parser.add_argument(
        '--summary-path',
        default='',
        help='Optional explicit summary.json path. Defaults to <out-dir>/summary.json.')
    return parser.parse_args()


def _parse_box_from_node(node):
    bndbox = node.find('bndbox')
    if bndbox is None:
        return None
    vals = []
    for key in ('xmin', 'ymin', 'xmax', 'ymax'):
        child = bndbox.find(key)
        if child is None or child.text is None:
            return None
        vals.append(float(child.text))
    return vals


def _node_name(node, default='Target'):
    name_node = node.find('name')
    if name_node is None or name_node.text is None:
        return default
    return name_node.text


def _target_nodes(root, tag, target_name):
    nodes = []
    for node in root.findall(tag):
        if _node_name(node, default=target_name) != target_name:
            continue
        box = _parse_box_from_node(node)
        if box is None:
            continue
        nodes.append((node, box))
    return nodes


def _box_contains(outer, inner):
    return (
        outer[0] <= inner[0] and
        outer[1] <= inner[1] and
        outer[2] >= inner[2] and
        outer[3] >= inner[3]
    )


def _make_cluster_node(name, box):
    cluster = ET.Element('cluster')
    name_node = ET.SubElement(cluster, 'name')
    name_node.text = str(name)
    bndbox = ET.SubElement(cluster, 'bndbox')
    for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), box):
        child = ET.SubElement(bndbox, key)
        float_value = float(value)
        if abs(float_value - round(float_value)) < 1e-6:
            child.text = str(int(round(float_value)))
        else:
            child.text = f'{float_value:.6f}'.rstrip('0').rstrip('.')
    return cluster


def _indent(tree):
    try:
        ET.indent(tree, space='  ')
    except AttributeError:
        pass


def build_cluster2(bbox_dir, cluster1_dir, out_dir, cluster_suffix, target_name):
    bbox_dir = Path(bbox_dir)
    cluster1_dir = Path(cluster1_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'bbox_dir': str(bbox_dir),
        'cluster1_dir': str(cluster1_dir),
        'out_dir': str(out_dir),
        'num_files': 0,
        'num_total_objects': 0,
        'num_original_clusters': 0,
        'num_added_clusters': 0,
        'num_final_clusters': 0,
        'num_files_with_additions': 0,
        'num_uncovered_before': 0,
        'num_uncovered_after': 0,
        'files_with_additions': [],
    }

    for bbox_path in sorted(bbox_dir.glob('*.xml')):
        img_id = bbox_path.stem
        cluster1_path = cluster1_dir / f'{img_id}{cluster_suffix}'
        bbox_root = ET.parse(bbox_path).getroot()
        if cluster1_path.is_file():
            cluster_root = ET.parse(cluster1_path).getroot()
        else:
            cluster_root = deepcopy(bbox_root)

        object_nodes = _target_nodes(bbox_root, 'object', target_name)
        cluster_nodes = _target_nodes(cluster_root, 'cluster', target_name)
        cluster_boxes = [box for _, box in cluster_nodes]

        summary['num_files'] += 1
        summary['num_total_objects'] += len(object_nodes)
        summary['num_original_clusters'] += len(cluster_nodes)

        added_count = 0
        uncovered_before = 0
        for obj_node, obj_box in object_nodes:
            covered = any(_box_contains(cluster_box, obj_box) for cluster_box in cluster_boxes)
            if covered:
                continue
            uncovered_before += 1
            cluster_root.append(_make_cluster_node(_node_name(obj_node, target_name), obj_box))
            cluster_boxes.append(list(obj_box))
            added_count += 1

        final_cluster_nodes = _target_nodes(cluster_root, 'cluster', target_name)
        uncovered_after = 0
        for _, obj_box in object_nodes:
            covered = any(_box_contains(cluster_box, obj_box) for _, cluster_box in final_cluster_nodes)
            if not covered:
                uncovered_after += 1

        summary['num_added_clusters'] += added_count
        summary['num_final_clusters'] += len(final_cluster_nodes)
        summary['num_uncovered_before'] += uncovered_before
        summary['num_uncovered_after'] += uncovered_after
        if added_count > 0:
            summary['num_files_with_additions'] += 1
            summary['files_with_additions'].append({
                'img_id': img_id,
                'num_objects': len(object_nodes),
                'original_clusters': len(cluster_nodes),
                'added_clusters': added_count,
                'final_clusters': len(final_cluster_nodes),
            })

        out_path = out_dir / f'{img_id}{cluster_suffix}'
        tree = ET.ElementTree(cluster_root)
        _indent(tree)
        tree.write(out_path, encoding='utf-8', xml_declaration=False)

    summary['files_with_additions'] = sorted(
        summary['files_with_additions'],
        key=lambda item: (-item['added_clusters'], item['img_id']))
    return summary


def main():
    args = parse_args()
    summary = build_cluster2(
        bbox_dir=args.bbox_dir,
        cluster1_dir=args.cluster1_dir,
        out_dir=args.out_dir,
        cluster_suffix=args.cluster_suffix,
        target_name=args.target_name,
    )
    summary_path = args.summary_path or osp.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'num_files': summary['num_files'],
        'num_total_objects': summary['num_total_objects'],
        'num_original_clusters': summary['num_original_clusters'],
        'num_added_clusters': summary['num_added_clusters'],
        'num_final_clusters': summary['num_final_clusters'],
        'num_files_with_additions': summary['num_files_with_additions'],
        'num_uncovered_before': summary['num_uncovered_before'],
        'num_uncovered_after': summary['num_uncovered_after'],
        'summary_path': summary_path,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
