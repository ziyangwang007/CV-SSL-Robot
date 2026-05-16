# Data

Put preprocessed EndoVis HDF5 files outside the repository, for example:

```text
../data/robotic/train/*.h5
../data/robotic/val/*.h5
../data/robotic/test/*.h5
```

Each HDF5 file should contain:

- `image`: RGB image shaped `H x W x 3`, or grayscale image shaped `H x W`
- `label`: integer mask shaped `H x W`

The `splits/` directory contains the train/validation/test split templates copied from the uploaded CV-WSL-Robot repository.
