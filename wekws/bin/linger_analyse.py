import torch
import torch.nn as nn 
import linger
import onnx

import argparse

import torch
import yaml

import onnx
import onnxruntime as ort

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wekws.utils.linger_utils import linger_quant, linger_export

def get_args():
    parser = argparse.ArgumentParser(description='export linger analyse')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--out_analyse', default="out_analyse.txt", help='out_analyse')
    parser.add_argument('--wb_analyse', default="wb_analyse.txt", help='wb_analyse')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn_file', required=True, help='cmvn file')
    parser.add_argument('--num_keywords', default=1, type=int, help='numbers of keywords')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # wb_analyse    
    linger.wb_analyse(args.checkpoint, args.wb_analyse)
    # out_analyse
    input_dim = configs['dataset_conf']['feature_extraction_conf'][
        'num_mel_bins']
    output_dim = args.num_keywords
    configs['model']['input_dim'] = input_dim
    configs['model']['output_dim'] = output_dim
    configs['model']['cmvn'] = {}
    configs['model']['cmvn']['norm_var'] = True
    configs['model']['cmvn']['cmvn_file'] = args.cmvn_file
    model = init_model(configs['model'])
    if "linger" in configs:
        model = linger_quant(model, configs["linger"])
    print(model)
    load_checkpoint(model, args.checkpoint)
    input_shape = configs['model']['linger']['input_shape']
    input_shape = [int(x) for x in input_shape.strip(',').split(',')]
    input = torch.randn(input_shape, dtype=torch.float) 
    cache = torch.zeros(1,
                        model.hdim,
                        model.backbone.padding,
                        dtype=torch.float)
    model.eval()
    with linger.Dumper() as dumper:
        dumper.analyse_layer_output(model)
        model(input, cache)
        dumper.save_out_analyse_log(save_log_path=args.out_analyse)
"""    
python ./test/linger_out.py \
--config ./conf/tcn_linger_float.yaml \
--checkpoint /data/user/yswang/task/wekws/exp/hi_xiaowen_tcn_linger_v1_float/avg_30.pt \
--cmvn_file /data/user/yswang/task/wekws/wekws/examples/hi_xiaowen/s0/data/train/global_cmvn
"""
if __name__ == '__main__':
    main()