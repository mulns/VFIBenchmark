# Video Frame Interpolation Benchmark

This is the repository for evaluating video frame interpolation (VFI) methods. These codes are used to produce metrics in "PerVFI".

In this repository, we include evaluation of both image datasets and video datasets. For image datasets, we support PSNR, SSIM, LPIPS, DISTS, and IE metrics. For video datasets, we support not only those metrics for image datasets, but also FloLPIPS and VFIPS.

## Usage

### Build models and metrics

1. Please download the model checkpoints into the `checkpoints` folder, and change the filepath in the [`build_models.py`](build_models.py) file.
2. Download the datasets to the `data/NAME-OF-DATASET` folder, and modify the corresponding dataloader in [`tools.py`](tools.py).
3. Run the scripts:

    Evaluate image datasets:
    ```bash
    python evaluate_imageDst.py \
        -m MODEL-NAME \
        -dst DATASET-NAME
    ```
    Evaluate video datasets:
    ```bash
    python evaluate_imageDst.py \
                -m MODEL-NAME \
                -dst DATASET-NAME
    ```

To locate supported `MODEL-NAME`, please refer to [`build_models.py`](build_models.py).

To locate supported `DATASET-NAME`, please refer to [`evaluate_imageDst.py`](evaluate_imageDst.py) and [`evaluate_videoDst.py`](evaluate_videoDst.py).


## Future:

This repository is under refinement. Please feel free to contribute or suggest improvements.