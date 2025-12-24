import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_list_images(folder_path, extensions={".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}):
    """
    List all image files in a folder.

    Args:
        folder_path (str): Path to the folder.
        extensions (set): Allowed image extensions (default: common formats).

    Returns:
        list[str]: List of image file paths.
    """
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in extensions
    ]

def plot_multiple(image_paths, image_labels=None,nrow=1,figsize=None):
    ncol = int(np.ceil(len(image_paths)/nrow))
    #print(ncol)
    if figsize is None:
        figsize=(ncol, nrow)
        
    fig, ax = plt.subplots(nrow,ncol, figsize=figsize)

    for id_image, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            img_now = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB) 
        else:
            img_now = image_path

        if nrow>1:
            #ax[id_image%nrow,id_image//nrow].imshow(img_now)
            #ax[id_image%nrow,id_image//nrow].axis("off")
            ax[id_image//nrow,id_image%nrow].imshow(img_now)
            ax[id_image//nrow,id_image%nrow].axis("off")
            if image_labels:
                #ax[id_image%nrow,id_image//nrow].set_title(image_labels[id_image])
                ax[id_image//nrow,id_image%nrow].set_title(image_labels[id_image])
        else:
            ax[id_image].imshow(img_now)
            ax[id_image].axis("off")
            if image_labels:
                ax[id_image].set_title(image_labels[id_image])
    
    plt.tight_layout()
    return fig,ax