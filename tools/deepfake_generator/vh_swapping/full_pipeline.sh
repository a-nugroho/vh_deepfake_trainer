
cd /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/
conda run --live-stream -n deepfake_dataset python run_attribute_json.py --json_path result_200k_live_face_dataset.json --dataset_name 200k_live_face_dataset --dir_images /mnt/ssd/datasets/deepfake/200k_live_face_dataset/live

# Generate e4s deepfakes
cd /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/e4s
conda run --live-stream -n faceswap_gen python scripts/face_swap_json.py --source_json /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/vh_swapping/collection/pair-200k_live_face_dataset-50+.json --output_dir /mnt/ssd/datasets/deepfake/200k_55plus/fake/e4s_2