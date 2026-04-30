import json
import os
import time
from glob import glob

import cv2
import pandas as pd
from face_detection import FaceDetection
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from face_deepfake import DeepfakeDetection

fd = FaceDetection()  # face detection wrapper
fdd = DeepfakeDetection(device_name="cuda")


class CSVDataset(Dataset):
    def __init__(self, csv_file, is_cropped=False):
        data = pd.read_csv(csv_file)
        base_paths = {
            "DFDC": "/mnt/SSD/dataset/deepfake/old_eval/DFDC/facebook_dfdc_frames/test/",
            "deepforensics": "/mnt/SSD/dataset/deepfake/old_eval/deeperforensics_1.0/DeeperForensics-1.0/extracted_frames",
            "Celeb-DF-v2": "/mnt/SSD/dataset/deepfake/old_eval/Celeb-DF-v2/Celeb-DF-v2/frames/test/",
            "faceforensics": "/mnt/SSD/dataset/deepfake/old_eval/faceforensics/cropped_images",
            "indonesian_deepfake_dataset_v2": "/mnt/SSD/dataset/deepfake/indonesian_deepfake_dataset_v2_test/",
        }

        dataset_name = next((name for name in base_paths if name in csv_file), None)

        self.image_paths = self._get_image_paths(data, dataset_name, base_paths)
        self.labels = data["label"].tolist()
        if dataset_name == "indonesian_deepfake_dataset_v2":
            self.labels = [0 if x == 1 else 1 for x in self.labels]
        self.is_cropped = is_cropped

    def _get_image_paths(self, data, dataset_name, base_paths):
        """Generate image paths based on the dataset name and path."""
        # Default case if dataset_name is not identified in the CSV
        if dataset_name is None:
            return data["image_path"].tolist()

        if dataset_name == "faceforensics":

            return [
                os.path.join(
                    base_paths[dataset_name],
                    os.path.join(x.split("/")[-2], os.path.basename(x)),
                )
                for x in data["image_path"].tolist()
            ]

        if dataset_name == "deepforensics":
            image_paths = []
            for path in data["image_path"].tolist():
                subdir = "real" if "real" in path else "deepfake"
                if subdir == "real":
                    image_paths.append(
                        os.path.join(
                            base_paths[dataset_name], "real", os.path.basename(path)
                        )
                    )
                else:
                    image_paths.append(
                        os.path.join(
                            base_paths[dataset_name],
                            "deepfake",
                            path.split("/")[-2],
                            os.path.basename(path),
                        )
                    )
            return image_paths

        try:
            return [
                os.path.join(base_paths[dataset_name], x)
                for x in data["image_path"].tolist()
            ]
        except KeyError:
            try:
                return [
                    os.path.join(base_paths[dataset_name], x)
                    for x in data["img_path"].tolist()
                ]
            except KeyError:
                return [
                    os.path.join(base_paths[dataset_name], x)
                    for x in data["path"].tolist()
                ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            return self.__getitem__(idx + 1)

        if not self.is_cropped:
            dets, ang = fd.predict(image)
            if len(dets) == 0:
                print(f"No face detected in {image_path}")
                return self.__getitem__(idx + 1)
            cropped_image, bbox = fd.crop_single_face_square(
                image, dets, ang, loose_factor=1.3, crop_size=None
            )
            image = cropped_image

        return image, label


class DirDataset(Dataset):
    def __init__(self, image_dir, is_cropped=False):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.is_cropped = is_cropped

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        label = 0 if "live" in image_path else 1  # Assign label based on filename

        if "bcad" in image_path or "raya" in image_path:
            label = 1
        elif "maybank" in image_path:
            label = 0

        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            return self.__getitem__(idx + 1)

        if not self.is_cropped:
            dets, ang = fd.predict(image)
            if len(dets) == 0:
                print(f"No face detected in {image_path}")
                return self.__getitem__(idx + 1)
            cropped_image, bbox = fd.crop_single_face_square(
                image, dets, ang, loose_factor=1.3, crop_size=None
            )
            image = cropped_image

        return image, label


# Dataset class to load data from a JSON file
class JSONDataset(Dataset):
    def __init__(self, json_file, is_cropped=False):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.image_paths = list(self.data.keys())
        self.labels = [self.data[path]["label"] for path in self.image_paths]
        self.processed_paths = [
            self.data[path]["processed_path"] for path in self.image_paths
        ]

        self.is_cropped = is_cropped

    def __len__(self):
        return len(self.image_paths)

    def get_preprocessed_path(self, image_path):
        """Generate a correct path for preprocessed images."""
        dataset_root = "/mnt/SSD/dataset/deepfake"

        if "UADFV" in image_path and "real" in image_path:
            image_path = image_path.replace("face/", "")
            return os.path.join(dataset_root, image_path)

        if os.path.isabs(image_path):
            return image_path

        return os.path.join(dataset_root, image_path)

    def __getitem__(self, idx):
        image_path = self.processed_paths[idx]
        label = self.labels[idx]

        image_path = self.get_preprocessed_path(image_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            return self.__getitem__(idx + 1)

        if not self.is_cropped:
            dets, ang = fd.predict(image)
            if len(dets) == 0:
                print(f"No face detected in {image_path}")
                return self.__getitem__(idx + 1)
            cropped_image, bbox = fd.crop_single_face_square(
                image, dets, ang, loose_factor=1.3, crop_size=None
            )
            image = cropped_image

        return image, label


def save_results_to_csv(results, output_csv):
    """Save inference results to a CSV file."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.DataFrame(
        results,
        columns=[
            "image_path",
            "actual_label",
            "predicted_label",
            "score",
            "processing_time",
        ],
    )
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def process_json_data(json_file, output_csv, is_cropped=False):
    """Process data from JSON file, load images, and perform inference."""
    dataset = JSONDataset(json_file, is_cropped)
    inference_and_save(dataset, output_csv, is_cropped)


def process_dir_data(image_dir, output_csv, is_cropped=False):
    """Process data from a directory of images, load and perform inference."""
    dataset = DirDataset(image_dir, is_cropped)
    inference_and_save(dataset, output_csv, is_cropped)


def process_csv_data(csv_file, output_csv, is_cropped=False):
    """Process data from a CSV file, load and perform inference."""
    dataset = CSVDataset(csv_file, is_cropped)
    inference_and_save(dataset, output_csv, is_cropped)


def inference_and_save(dataset, output_csv, is_cropped=False):
    """Perform inference on dataset and save results, also check feature importancXe."""

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    for idx, batches in tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Processing Images"
    ):
        batch_image_paths = dataset.image_paths[
            idx * data_loader.batch_size : (idx + 1) * data_loader.batch_size
        ]

        images, labels = batches
        start_time = time.time()
        image = images[0].detach().cpu().numpy()
        scores = fdd.predict(image)
        pred_labels = fdd.classify_predictions(scores)
        processing_time = time.time() - start_time

        results.append(
            [
                batch_image_paths[0],
                labels[0].item(),
                int(pred_labels),
                scores,
                processing_time,
            ]
        )

    save_results_to_csv(results, output_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="JSON file path")
    parser.add_argument("--dir", type=str, help="Directory path")
    parser.add_argument("--csv", type=str, help="CSV file path")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--is_cropped", action="store_true", help="Use cropped images")
    args = parser.parse_args()

    if args.json:
        process_json_data(args.json, args.output, args.is_cropped)
    elif args.dir:
        process_dir_data(args.dir, args.output, args.is_cropped)
    elif args.csv:
        process_csv_data(args.csv, args.output, args.is_cropped)
