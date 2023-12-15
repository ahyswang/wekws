import numpy as np 
import onnx
from onnx import numpy_helper
import math 
import sys 
import os 
import zlib

def get_node(graph, name):
    for node in graph.node:
        if node.name == name:
            return node
    return None

def get_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            #return onnx.numpy_helper.to_array(init)
            # fix onnx 1.7.0
            return numpy_helper.to_array(init)
    return None 

def get_attribute(node, name):
    for a in node.attribute:
        value = None 
        if a.name == name:
            for f in ["f", "i", "s"]:
                if a.HasField(f):
                    value = getattr(a, f)
                    # Needed for supporting python version > 3.5
                    if isinstance(value, bytes):
                        value = value.decode(encoding="utf-8")
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    value = tuple(getattr(a, f))
            return value 
    return None

def parse_onnx(onnx_path):
    """
    parse onnx
    """
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    param_array = []

    # conv1d_1 = get_node(graph, ".root")
    # scale_x = get_attribute(conv1d_1, "scale_x")
    weight = get_initializer(graph, "global_cmvn.mean")
    bias = get_initializer(graph, "global_cmvn.istd")
    print(weight.shape)
    print(bias.shape)

"""
operator
{
    attr
    forward(*tensors, tensors_num)
}

operator list
tensor list
refcount[opnum] : op_id => int
outputs[opnum] : op_id => tensor

{
    { "id":0, "name":"null", "inputs":[], "attr":{ }},
    { "id":1, "name":"linear", "inputs":[0], "attr":{ }},
    { "id":2, "name":"add", "inputs":[1,0], "attr":{ }},
    { "id":3, "name":"linear", "inputs":[2], "attr":{ }},
    { "id":4, "name":"add", "inputs":[3,2], "attr":{ }},
}

"""

class Tensor(object):
    def __init__(self, shape=(), data=None, dtype=0):
        super().__init__()
        self.realloc(shape, data, dtype)
    def realloc(self, shape=(), data=None, dtype=0):
        assert type(shape) == tuple
        assert data == None or len(data) == sum(shape)
        self.shape = shape 
        self.dtype = dtype
        self.size  = sum(shape)
        if data != None:
            self.data = data
        else:
            self.data = [0] * self.size
    def __str__(self):
        return "{shape:%s, dtype:%s, data:%s, size:%s}"%(self.shape, self.dtype, self.data, self.size)

class BaseOP(object):
    def __init__(self, attr={}):
        super().__init__()
        print("attr", attr)
        assert "id" in attr
        assert "name" in attr
        assert "op" in attr
        assert "inputs" in attr
        self.attr = attr
        self.id = attr["id"]
        self.name = attr["name"]
        self.op = attr["op"]
        self.inputs = attr["inputs"]
    def forward(self, inputs:tuple) -> Tensor :
        print("BaseOP forward")
        assert False, "unsport!!!!"
    def __str__(self):
        return "{id:%s,name:%s,op:%s}"%(self.id, self.name, self.op)

class Add(BaseOP):
    def __init__(self, attr={}):
        super().__init__(attr)
    def forward(self, inputs:tuple) -> Tensor :
        assert len(inputs) == 2
        assert inputs[0].size == inputs[1].size
        x1 = inputs[0]
        x2 = inputs[1]
        y1 = Tensor(x1.shape)
        for i in range(x1.size):
            y1.data[i] = x1.data[i] + x2.data[i]
        return y1

class Null(BaseOP):
    def __init__(self, attr={}):
        super().__init__(attr)
    def forward(self, inputs:tuple) -> Tensor :
        assert len(inputs) == 1
        x1 = inputs[0]
        y1 = Tensor(x1.shape)
        for i in x1.size:
            y1.data[i] = x1.data[i]
        return y1

class Concat(BaseOP):
    def __init__(self, attr={}):
        super().__init__(attr)
        self.attr = attr["attr"]
        assert "axis" in attr["attr"]
        self.axis = attr["attr"]["axis"]
    def forward(self, inputs:tuple) -> Tensor :
        assert len(inputs) == 2
        x1 = inputs[0]
        x2 = inputs[1]
        x1 = np.array(x1.data).reshape(x1.shape)
        x2 = np.array(x2.data).reshape(x2.shape)
        y1 = np.concatenate((x1, x2), self.axis)
        y1 = y1.tolist()
        # y1 = Tensor((x1.shape[0] + x2.shape[1]))
        # for i in range(y1.size):
        #     if i >= x1.size():
        #         y1.data[i] = x2.data[i-x1.size]
        #     else:
        #         y1.data[i] = x1.data[i] 
        return y1

class Linear(BaseOP):
    def __init__(self, attr={}):
        super().__init__()
        self.attr = attr
    def forward(self, inputs, outputs):
        pass
        print("Linear forward")

op_map = {
    "null" : lambda attr: Null(attr),
    "add" : lambda attr: Add(attr),
    "concat" : lambda attr: Concat(attr), 
    "linear" : lambda attr: Linear(attr), 
}
    
class Excute(object):
    def __init__(self, attrs_list):
        super().__init__()
        self.attrs_list = attrs_list
        self.op_list = []
        self.outputs = [] 
        self.refs = [0] * len(attrs_list)
        for attr in attrs_list:
            assert "name" in attr
            assert "id" in attr
            assert "op" in attr
            assert "inputs" in attr
            assert attr["op"] in op_map
            for input_id in attr["inputs"]:
                assert input_id < len(self.op_list)
                assert input_id < len(self.refs)
                self.refs[input_id] += 1
            self.op_list.append(op_map[attr["op"]](attr))
            self.outputs.append(None)

    def set_input(self, id, input):
        assert id >= 0 and id < len(self.outputs)
        self.outputs[id] = input

    def get_output(self, id):
        assert id >= 0 and id < len(self.outputs)
        return self.outputs[id]
    
    def forward(self, start=0, end=1000000):
        for op in self.op_list:
            if op.id < start:
                continue
            if op.id >= end:
                break

            input_ids = op.inputs
            inputs = []
            for id in input_ids:
                inputs.append(self.outputs[id])
            import pdb; pdb.set_trace()
            self.outputs[op.id] = op.forward(inputs)

            for id in input_ids:
                assert id < len(self.outputs)
                assert self.refs[id] > 0
                self.refs[id] -= 1
                if 0==self.refs[id]:
                    print("[DEBUG] free output, id:%d"%(id))
                    self.outputs[id] = None 

    def __str__(self):
        s = "op_list:[\n"
        for op in self.op_list:
            s += str(op) + ",\n"
        s += "],\n"
        s += "refs:" + str(self.refs) + ",\n"
        return s

op_list = [ { "id":0, "name":"null", "op":"null", "inputs":[], "attr":{ }},
            { "id":1, "name":"null", "op":"null", "inputs":[], "attr":{ }},
           # { "id":2, "name":"add", "op":"add", "inputs":[1,0], "attr":{ }},
            { "id":2, "name":"add", "op":"concat", "inputs":[1,0], "attr":{ "axis":0}} ]
     
def test_op_1():
    op1 = BaseOP({"id":0, "name":"op1", "op":"BaseOP","inputs":[]})
    #op1.forward((), ())
    print(str(op1))
    op1 = Add({"id":0, "name":"op1", "op":"Add","inputs":[]})
    x1 = Tensor((2,), [1,2])
    x2 = Tensor((2,), [5,6])
    x3 = Tensor((2,), [0,0])
    x3 = op1.forward((x1, x2))
    print("Add(x1,x2), x3 =", x3)

def test_exec_1():
    x1 = Tensor((2,), [1,2])
    x2 = Tensor((2,), [5,6])
    exec = Excute(op_list)
    exec.set_input(0, x1)
    exec.set_input(1, x2)
    print(exec)
    exec.forward(2)
    x3 = exec.get_output(2)
    print("Exec, x3 =", x3)
 
"""
outputs = [o0, o1, o2]; => data offset
inits = [i0, i1]; => data offset
tensors = [i0, i1, o0, o1, o2]

"""
if __name__ == "__main__":

    # onnx_path = "/data/user/yswang/task/wekws/exp/hi_xiaowen_tcn_linger_v1/avg_1.linger.simplify.onnx"
    # parse_onnx(onnx_path)
    test_op_1()
    test_exec_1()
