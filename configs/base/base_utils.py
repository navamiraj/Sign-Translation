from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = "checkpoint_pose_gcn"
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "data"
    return lmdb_path
