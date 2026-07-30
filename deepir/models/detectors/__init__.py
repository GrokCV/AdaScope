# Copyright (c) GrokCV. All rights reserved.

__all__ = []


def _safe_export(module_name, symbols):
    try:
        module = __import__(f'{__name__}.{module_name}', fromlist=list(symbols))
    except ModuleNotFoundError:
        return

    for symbol in symbols:
        if hasattr(module, symbol):
            globals()[symbol] = getattr(module, symbol)
            __all__.append(symbol)


_safe_export('cluster', ['Cluster'])
_safe_export('yolc', ['YOLC'])
_safe_export('flat_sync_clean_grpo_detector', ['FlatSyncCleanGRPODetector'])
_safe_export('flat_sync_clean_grpo_detector_visdrone', ['FlatSyncCleanGRPODetectorVisDrone'])
_safe_export('sync_grpo_detector', ['SyncGRPODetector'])
_safe_export('sync_clean_grpo_detector', ['SyncCleanGRPODetector'])
_safe_export('sync_clean_grpo_detector_visdrone', ['SyncCleanGRPODetectorVisDrone'])
_safe_export(
    'syn_single_stage_rawcluster_grpo_detector',
    ['SynSingleStageRawClusterGRPODetector'],
)
_safe_export(
    'syn_single_stage_rawcluster_fpnroi_fcos_detector',
    ['SynSingleStageRawClusterFPNRoIFCOSDetector'],
)
_safe_export(
    'syn_single_stage_rawcluster_fpnroi_grpo_detector',
    ['SynSingleStageRawClusterFPNRoIGRPODetector'],
)
