import os

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