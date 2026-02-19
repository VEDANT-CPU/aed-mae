import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 32
    config.eval_freq = 5
    config.epochs = 50
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "/nfs_home/users/poonam/vedant_project_dataset/aed-mae/results_r1"  # the checkpoints will be loaded from here
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = "/nfs_home/users/poonam/vedant_project_dataset/aed-mae/results_r1/checkpoint-latest.pth"
    #config.resume = False
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-5

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "/nfs_home/users/poonam/divya/anomaly_datasets/fall_detect/frames"
    config.avenue_gt_path = "/nfs_home/users/poonam/divya/anomaly_datasets/fall_detect/Fall_gt"
    config.percent_abnormal = 0.0 #0 means ignore the abnormal folders.
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 1
    config.num_workers = 4
    config.pin_mem = False

    return config
