This tool is used to analyze available dataset, and create a selective compilation dataset.

The compilation dataset efficiently select representative demographic of face profile, source, and deepfake methods to optimize training process.

The objective is to:

1. Minimize overfit by ensuring challenging-to-learn training set, and reliable separate validation set.
2. Train on multiple epochs to get multiple cycles of same data points with different augmentations.
3. Higher variety of cropping area
4. Extract multiple sets of features using different augmentations to enable off-line extraction for stage 2 training, e.g. model ensembling.

The profile includes:
1. Age
2. Gender
- Male
- Female
3. Accessory (Hijab)
4. Image quality (sharpness)
- Sharp
- High res
- Low res
- Blurry
5. Deepfake smoothness (e.g. Passed through the ConvNext model?)
- Easy Negatives (Model 1, Model 2, Mix)
- Hard Negatives (Model 1, Model 2, Mix)
- Easy Positives (Model 1, Model 2, Mix)
- Hard Positives (Model 1, Model 2, Mix)

Deepfake methods:
1. Inswapper Low-res
2. Inswaper hi-res
3. ReSwapper hi-res


Live sources:
1. 200k-dataset

Public Dataset:
1. DFDC
2. DF40

Categorical
1. VH DF-40 Deepfake. Inspired by DF40, reach the same number of samples with more variety of deepfake and live sources. 100,000 images (?)