import torch
import torch.nn as nn 
import linger
import onnx
from onnxsim import simplify

def linger_quant(model, configs):
    if "clamp" == configs["stage"] or "quant" == configs["stage"]:
        print("linger clamp")
        #import pdb; pdb.set_trace()
        input_shape = configs['input_shape']
        input_shape = [int(x) for x in input_shape.strip(',').split(',')]
        input = torch.zeros(input_shape) 
        linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
        linger.trace_layers(model, model, (input, ), fuse_bn=True)
        linger.disable_normalize(model.global_cmvn)
        # linger.disable_normalize(model.preprocessing)
        # linger.disable_normalize(model.classifier)
        #type_modules = (nn.Conv2d, linger.NormalizeConvBN2d)
        #linger.normalize_module(model.fc, type_modules=type_modules, normalize_weight_value=8, normalize_bias_value=None, normalize_output_value=2) 
        #import pdb; pdb.set_trace();
        normalize_modules = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Embedding)
        model.backbone = linger.normalize_layers(model.backbone, normalize_modules=normalize_modules, normalize_weight_value=8, normalize_bias_value=None, normalize_output_value=8)
    if "quant" == configs["stage"]:
        print("linger quant")
        linger.SetFunctionBmmQuant(True)
        #linger.disable_quant(model.fc)
        linger.disable_quant(model.global_cmvn)
        # linger.disable_quant(model.preprocessing)
        # linger.disable_quant(model.classifier)
        linger.SetIQTensorSigmoid(False)
        quant_modules = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.Embedding)
        model = linger.init(model, quant_modules=quant_modules, mode=linger.QuantMode.QValue)
    return model

def linger_export(model, configs, onnx_path):
    assert "quant" == configs["stage"]
    model.eval()
    input_shape = configs['input_shape']
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

def linger_dump(model, configs):
    model.eval()
    assert "quant" == configs["stage"]
    input_shape = configs['input_shape']
    input_shape = [int(x) for x in  input_shape.split(',')]
    input = torch.randn(input_shape) 
    dummy_input = torch.randn(input_shape, dtype=torch.float)
    cache = torch.zeros(1,
                        model.hdim,
                        model.backbone.padding,
                        dtype=torch.float)
    with linger.Dumper() as dumper:
        dumper.enable_dump_quanted(model, path=configs["dump_dir"])
        output = model(input, cache)

def linger_wb_analyse(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint = checkpoint["state_dict"]
    linger.wb_analyse(checkpoint, "wb_anylse.log")

def linger_out_analyse(model, configs):
    model.eval()
    assert "quant" == configs["stage"]
    input_shape = configs['input_shape']
    input_shape = [int(x) for x in  x.split(',')]
    input = torch.randn(input_shape) 
    dummy_input = torch.randn(input_shape, dtype=torch.float)
    cache = torch.zeros(1,
                        model.hdim,
                        model.backbone.padding,
                        dtype=torch.float)
    with linger.Dumper() as dumper:
        dumper.analyse_layer_output(model, match_pattern="root.")
        model(input, cache)
        dumper.save_out_analyse_log(save_log_path="out_anylse.log")