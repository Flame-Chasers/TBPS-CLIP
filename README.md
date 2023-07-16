# An Empirical Study of CLIP for Text-based Person Search

This repository is the code for the paper [An Empirical Study of CLIP for Text-based Person Search]().

### Environment

The required packages are listed in `requirements.txt`. You can install them using:

```sh
pip install -r requirements.txt
```

### Configuration

The settings, including the checkpoint, training schedule, hyperparameters, etc., can be modified in `config.yaml`.

### Training

You can start the training using PyTorch's torchrun with ease:

```sh
TORCHELASTIC_ERROR_FILE=error.log \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
main.py
```

### Citation
If our work helps, please consider citing:

```

```

### License
This code is distributed under an MIT LICENSE.
