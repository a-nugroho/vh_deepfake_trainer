from face_detection import FaceDetection
import cv2
import os
from pathlib import Path

# Initialize face detection with GPU and warm-up
fd = FaceDetection(use_cuda=True)


root_output = "/mnt/ssd/datasets/deepfake/vh_55plus/fake/"
faceswap_method = ""
# Read JSON Here

# Read source and target pair to ensure face bbox is valid
# Load an image
path_source = key_now
img = cv2.imread(path_source)
valid_source = False
dets, angle = fd.predict(img, strict_level="medium")
if len(dets)>0:
    valid_source = True

path_target = value_now["swap_pair"]
img = cv2.imread(path_target)
valid_target = False
dets, angle = fd.predict(img, strict_level="medium")
if len(dets)>0:
    valid_target = True

if valid_source & valid_target:
    # Perform face swap
    name_output = Path(path_source).stem + "--to--" + Path(path_target).stem
    path_output = os.path.join(root_output,faceswap_method,name_output+".jpg")

# E4S model
net = Net3(opts)
net = net.to(opts.device)
save_dict = torch.load(opts.checkpoint_path)
net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
net.latent_avg = save_dict['latent_avg'].to(opts.device)
print("Load E4S pre-trained model success!") 
# ========================================================  

if len(opts.target_mask)!= 0:
    target_mask = Image.open(opts.target_mask).convert("L")
    target_mask_seg12 = __celebAHQ_masks_to_faceParser_mask_detailed(target_mask)
else:
    target_mask_seg12 = None

list_source_path = #get from json
list_target_path = #get from json
for source_now in list_source_path:
    for target_now in list_target_path:
        try:
            faceSwapping_pipeline(source_now, target_now, opts, save_dir=opts.output_dir, 
                                target_mask = target_mask_seg12, need_crop = True, verbose = opts.verbose) 
        except:
            continue    
