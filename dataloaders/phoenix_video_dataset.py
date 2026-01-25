from torch.utils.data import Dataset
import lmdb
from PIL import Image
import io
import pickle
import numpy as np
import torch
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from tqdm import tqdm
import pandas as pd
import ignite.distributed as idist
import cv2
import os

YOLO_BOX_DIR = "/data/sign2gpt/Sign2GPT/data/phoenix2014t/yolo_boxes"

class PhoenixVideoDataset(Dataset):
    def __init__(
        self,
        df,
        lmdb_video_dir,
        dict_gloss_to_id,
        transform,
        isValid,
        dict_sentence=None,
        dict_lem_to_id=None,
        dict_lem_counter=None,
    ):
        self.items = df.to_dict("records")
        self.lmdb_dir = lmdb_video_dir
        self.transform = transform
        self.isValid = isValid
        self.lmdb_util_video = None
        self.dict_gloss_to_id = dict_gloss_to_id
        self.dict_sentence = dict_sentence
        self.dict_lem_to_id = dict_lem_to_id
        self.dict_lem_counter = dict_lem_counter
        self.use_yolo_boxes = True
        self.yolo_box_dir = YOLO_BOX_DIR
        self.boxes_cache = {}

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}

    def __getitem__(self, idx):
        item = self.items[idx]
        file_name = item["name"]
        if self.use_yolo_boxes and file_name not in self.boxes_cache:
            box_path = os.path.join(self.yolo_box_dir, f"{file_name}.npy")
            self.boxes_cache[file_name] = np.load(box_path, allow_pickle=True).item()

        if not self.lmdb_util_video:
            self.lmdb_util_video = LMDBUtility(
                f"{self.lmdb_dir}/{file_name}",
            )
            self.num_frames = self.lmdb_util_video.details["num_frames"]

        if self.isValid:
            start_frame = 0
            end_frame = self.num_frames
        else:
            start_frame = np.random.randint(0, self.transform.random_shift)
            end_frame = np.random.randint(
                self.num_frames - self.transform.random_shift, self.num_frames + 1
            )

        selection = np.arange(start_frame, end_frame, self.transform.stride)

        selection = selection.astype(int)

        if len(selection) > self.transform.max_seq_len:
            selection = np.random.choice(
                selection, size=self.transform.max_seq_len, replace=False
            )

            selection = np.sort(selection)

        frames = self.lmdb_util_video.get_frames(selection)

        processed_frames = []
        
        for frame_idx, frame in zip(selection, frames):
            person_box = self.boxes_cache[file_name].get(frame_idx)
            h, w, _ = frame.shape
            if person_box is None:
                side = min(h, w)
                y1 = (h - side) // 2
                x1 = (w - side) // 2
                crop = frame[y1:y1 + side, x1:x1 + side]
                crop_224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                x1, y1, x2, y2 = person_box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                crop_224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    
            processed_frames.append(crop_224)
        
        frames = processed_frames
        if self.transform:
            frames = self.transform.aug_video(frames, self.isValid)
        else:
            frames = torch.tensor(np.stack(frames)).float()
        sentence = item["translation"]
        glosses = item["orth"].split(" ")
        pseudo_gloss_ids = []
        if self.dict_sentence is not None:
            if sentence in self.dict_sentence:
                lems = self.dict_sentence[sentence]
                pseudo_gloss_ids = [
                    self.dict_lem_to_id[lem]
                    for lem in lems
                    if self.dict_lem_counter[lem] / len(self.dict_sentence) < 0.4
                ]
                # pseudo_gloss_ids.append(self.dict_lem_to_id[lem])

        if self.dict_gloss_to_id:
            gloss_ids = [self.dict_gloss_to_id[gloss] for gloss in glosses]

        return {
            "index": torch.tensor(idx).long(),
            "frames": frames,
            "sentence": sentence,
            "file_name": file_name,
            **(
                {"gloss_ids": torch.tensor(gloss_ids).long()}
                if self.dict_gloss_to_id
                else {}
            ),
            **(
                {"pseudo_gloss_ids": torch.tensor(pseudo_gloss_ids).long()}
                if self.dict_lem_to_id
                else {}
            ),
        }


def get_ds(ds_params, transform):
    from dataloaders.data_utils.file_utils import read_json, read_pickle

    df = pd.read_csv(ds_params["csv_dir"], sep=ds_params["sep"])
    if "gloss_dir" in ds_params:
        dict_gloss_to_id = read_json(ds_params["gloss_dir"])
    else:
        dict_gloss_to_id = None

    if "pseudo_gloss_dir" in ds_params:
        dict_processed_words = read_pickle(ds_params["pseudo_gloss_dir"])

        dict_sentence = dict_processed_words["dict_sentence"]
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence = None
        dict_lem_to_id = None
        dict_lem_counter = None
    list_of_ds = []
    length = 0
    for gp, d in tqdm(df.groupby("name")):
        ds = PhoenixVideoDataset(
            d,
            **ds_params["ds_params"],
            dict_gloss_to_id=dict_gloss_to_id,
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            transform=transform,
        )
        list_of_ds.append(ds)
        collate_fn = ds.collate_fn
        length += len(ds)

    ds = MyConcatDataset(list_of_ds)
    dl = idist.auto_dataloader(
        ds,
        shuffle=ds_params["shuffle"],
        sampler=None,
        num_workers=ds_params["num_workers"],
        batch_size=ds_params["bs"],
        drop_last=ds_params["drop_last"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dl, {"length": length, "dict_lem_to_id": dict_lem_to_id}
