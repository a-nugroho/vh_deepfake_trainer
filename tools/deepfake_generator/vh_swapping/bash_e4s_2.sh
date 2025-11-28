cd /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/e4s
CUDA_VISIBLE_DEVICES=0 conda run --live-stream -n faceswap_gen python scripts/face_swap_json.py --source_json /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/vh_swapping/collection/pair-200k_live_face_dataset-24nov.json --output_dir /mnt/ssd/datasets/deepfake/200k_24nov/fake/e4s

#conda run --live-stream -n faceswap_gen python scripts/face_swap_json.py --source_json /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/vh_swapping/collection/pair-200k_live_face_dataset-50+.json --output_dir /mnt/ssd/datasets/deepfake/200k_55plus/fake/e4s_20251103


#cd /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/MobileFaceSwap
#conda run --live-stream -n faceswap_gen python face_swap_json.py --source_json /mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/deepfake_generator/vh_swapping/collection/pair-200k_live_face_dataset-50+.json --output_dir /mnt/ssd/datasets/deepfake/200k_55plus/fake/mobilefaceswap