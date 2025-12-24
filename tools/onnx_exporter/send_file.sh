PORT_TARGET='2022'
#source_orig_path=/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/training/logs/training/effort_2025-12-18-07-29-00/train/training_indonesian_deepfake_dataset_v2_updated,df40_train_fs_official,facebook_dfdc_train_reduced_rev,test_200k_24nov_merged_train/ckpt_latest
#source_path=${source_orig_path}_iter277000.pth
source_orig_path=/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/training/logs/training/effort_2025-12-24-07-52-16/train/training_indonesian_deepfake_dataset_v2_updated,df40_train_fs_official,facebook_dfdc_train_reduced_rev,test_200k_24nov_merged_train,reswapper_v2_train/ckpt_latest
source_path=${source_orig_path}_iter18000.pth
target_addr=a.nugroho@10.8.0.49
target_path=/mnt/HDD/workspace/adi/repos/vh_deepfake_trainer/tools/onnx_exporter/checkpoints/effort_2025-12-24-07-52-16/
cp ${source_orig_path}.pth ${source_path} 
rsync -avz -e 'ssh -i ~/.ssh/id_ed25519 -p '$PORT_TARGET ${source_path} $target_addr:$target_path