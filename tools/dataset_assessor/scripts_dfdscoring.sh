#CUDA_VISIBLE_DEVICES=1 python run_attribute_json_fd-3-1-0.py --json_path result_200k_live_face_dataset.json --dataset_name 200k_live_face_dataset --dir_images /mnt/ssd/datasets/deepfake/200k_live_face_dataset/live


#CUDA_VISIBLE_DEVICES=1 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/test_200k_merged_train.json --dataset_name test_200k_merged_train

#CUDA_VISIBLE_DEVICES=1 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/df40_train_fs_reduced.json --dataset_name df40_train_fs_reduced

#CUDA_VISIBLE_DEVICES=0 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/facebook_dfdc_train_reduced_rev.json --dataset_name facebook_dfdc_train_reduced_rev

#CUDA_VISIBLE_DEVICES=0 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/facebook_dfdc_test_rev.json --dataset_name facebook_dfdc_test_rev

CUDA_VISIBLE_DEVICES=0 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/df40_train_fsonly_rev.json --dataset_name df40_train_fsonly_rev

#CUDA_VISIBLE_DEVICES=0 python run_dfd_scoring.py --json_path /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json/test_200k_24nov_merged_train.json --dataset_name test_200k_24nov_merged_train
