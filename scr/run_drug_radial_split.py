#!/usr/bin/env python

import train


CONFIG = {
    "cuda_device": 0,
    "data_name": "SNAP",
    "split_mode": "drug_morgan_distance_radial_split",
    "ratio_dir": "1_1",
    "group_ids": range(1, 6),
    "batch_size": 32,
    "lr": 5e-5,
    "epochs": 100,
    "patience": 10,
    "early_stop_start_epoch": 6,
    "result_file": "summary.xlsx",
}


if __name__ == "__main__":
    train.main(CONFIG)
