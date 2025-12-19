# Font Classifier

## Train your own model

### Getting started
Simply install all necessary python dependencies in a virtual environment:
```
python -m venv font-venv
source font-venv/bin/activate
pip install -r requirements.txt
```

### Run training
To run training, we recommend a GPU. However, it is feasible to run on CPU as well.
```
python train.py --image_folder=... --output_folder=...
```
By default, this will train a Resnet50 model, but you can easily swap a different architecture by setting the `--network_type` flag to one of the network types supported by the [timm library](https://huggingface.co/docs/timm/en/reference/models).

### Run inference
Finally, you can use the inference script on your own model:
```
python infer.py --model_folder=... --data_folder=...
```
