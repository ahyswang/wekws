import torch
import torch.nn as nn 
import linger

def linger_clamp(model, configs):
    input_dim = configs['model']['input_dim']
    input = torch.zeros((1, 100, input_dim)) 
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    linger.trace_layers(model, model, (input, ), fuse_bn=True)
    #linger.disable_normalize(model)
    #type_modules = (nn.Conv2d, linger.NormalizeConvBN2d)
    #linger.normalize_module(model.fc, type_modules=type_modules, normalize_weight_value=8, normalize_bias_value=None, normalize_output_value=2) 
    #import pdb; pdb.set_trace();
    normalize_modules = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Embedding)
    model = linger.normalize_layers(model, normalize_modules=normalize_modules, normalize_weight_value=8, normalize_bias_value=None, normalize_output_value=8)
    return model

def linger_quant(model, configs):
    linger.SetFunctionBmmQuant(True)
    #linger.disable_quant(model.fc)
    quant_modules = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.Embedding)
    model = linger.init(model, quant_modules=quant_modules, mode=linger.QuantMode.QValue)
    return model

def quant_model(model, configs):
    quant = configs.get('quant', '')
    quant_conf = configs.get('quant_conf', {})
    if "linger" == quant:
        if 1 == quant_conf["stage"]:
            model = linger_clamp(model, configs)
        if 2 == quant_conf["stage"]:
            model = linger_clamp(model, configs)
            model = linger_quant(model, configs)
    return model

