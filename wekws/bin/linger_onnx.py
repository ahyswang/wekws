# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import torch
import yaml

import onnx
import onnxruntime as ort

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wekws.utils.linger_utils import linger_quant, linger_export
from onnxsim import simplify
from wekws.utils.train_utils import count_parameters, set_mannul_seed
import linger
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='export to onnx model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--onnx_model',
                        required=True,
                        help='output onnx model')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    set_mannul_seed(100)
    feature_dim = configs['model']['input_dim']
    hidden_dim = configs['model']['hidden_dim']
    model = init_model(configs['model'])
    if configs['training_config'].get('criterion', 'max_pooling') == 'ctc':
        # if we use ctc_loss, the logits need to be convert into probs
        model.forward = model.forward_softmax
    print(model)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    
    onnx_path = args.onnx_model
    # save mean/istd.
    mean = model.global_cmvn.mean.numpy()
    istd = model.global_cmvn.istd.numpy()
    meanistd = np.concatenate((mean, istd), axis=0)
    meanistd.tofile(onnx_path.replace("onnx","meanistd.bin"))

    model.eval()
    model.onnx = True
    input_shape = configs["model"]["linger"]['input_shape']
    input_shape = [int(x) for x in input_shape.strip(',').split(',')]
    input = torch.randn(input_shape) 
    dummy_input = torch.randn(input_shape, dtype=torch.float)
    cache = torch.zeros(1,
                        model.hdim,
                        model.backbone.padding,
                        dtype=torch.float)
    
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        torch.onnx.export(model, (dummy_input, cache),
                        onnx_path,
                        input_names=['input', 'cache'],
                        output_names=['output', 'r_cache'],
                            dynamic_axes={
                                'input': {
                                    1: 'T'
                                },
                                'output': {
                                    1: 'T'
                                }},
                        opset_version=11,
                        verbose=False,
                        export_params=True,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # simplyfy
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, onnx_path.replace("onnx", "simplify.onnx"))

    # dump
    with linger.Dumper() as dumper:
        dumper_dir = onnx_path.replace("onnx", "dumper_tensor")
        if not os.path.exists(dumper_dir):
            os.makedirs(dumper_dir)
        dumper.enable_dump_quanted(model, path=dumper_dir)
        output, r_cache = model(dummy_input, cache)
        dummy_input.detach().numpy().tofile(dumper_dir+"/dummy_input.bin")
        cache.detach().numpy().tofile(dumper_dir+"/cache.bin")
        output.detach().numpy().tofile(dumper_dir+"/output.bin")
        r_cache.detach().numpy().tofile(dumper_dir+"/r_cache.bin")

if __name__ == '__main__':
    main()
