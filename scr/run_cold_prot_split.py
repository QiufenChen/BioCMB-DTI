import train


CONFIG = {
    "cuda_device": 0,
    "split_mode": "cold_prot_split",
    "data_name": "SNAP",
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
