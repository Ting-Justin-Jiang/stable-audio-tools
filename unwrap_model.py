import argparse
import json
import torch
from torch.nn.parameter import Parameter
from stable_audio_tools.training.factory import create_training_wrapper_from_config
from stable_audio_tools.models import create_model_from_config

_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model-config', type=str, default=None)
    args.add_argument('--ckpt-path', type=str, default=None)
    args.add_argument('--name', type=str, default='exported_model')
    args.add_argument('--use-safetensors', action='store_true')

    args = args.parse_args()

    with open(args.model_config) as f:
        model_config = json.load(f)
    
    model = create_model_from_config(model_config)
    
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)

    # Always build the correct training wrapper from the config, then load the checkpoint's state_dict
    assert args.ckpt_path is not None, 'ckpt-path must be provided'
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    if args.ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load_file
        state_dict = safe_load_file(args.ckpt_path)
    else:
        ckpt_obj = torch.load(args.ckpt_path)
        state_dict = ckpt_obj.get('state_dict', ckpt_obj)

    training_wrapper.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded model from {args.ckpt_path}")

    if args.use_safetensors:
        ckpt_path = f"{args.name}.safetensors"
    else:
        ckpt_path = f"{args.name}.ckpt"

    training_wrapper.export_model(ckpt_path, use_safetensors=args.use_safetensors)

    print(f"Exported model to {ckpt_path}")