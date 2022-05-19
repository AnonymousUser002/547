from copy import deepcopy
import difflib
# from tokenize import String
from utils.utils import *

'''This is the corresponding supported list for known non-supported operators'''
supported_list = [            
    "Add","AveragePool","BatchNormalization","Cast","Ceil","Clip","Concat","Constant","ConstantOfShape","Conv","ConvTranspose","Div","Elu","Equal","Erf","Exp","Expand","Flatten","Floor","Gather","GatherND","Gemm","GlobalAveragePool","Greater","Identity","InstanceNormalization","LeakyRelu","Less","Log","Loop","LSTM","MatMul","Max","MaxPool","Min","Mul","NonMaxSuppression","Not","OneHot","Or","Pad","Pow","PRelu","Range","Reciprocal","ReduceMean","ReduceProd","ReduceSum","Relu","Reshape","Resize","Scatter","ScatterElements","ScatterND","Shape","Sigmoid","Slice","Softmax","Softplus","Softsign","Split","Sqrt","Squeeze","Sub","Tanh","ThresholdedRelu","Tile","TopK","Transpose","Unsqueeze","Upsample","Where","LpNormalization","SpaceToDepth","DepthToSpace","HardSwish","Einsum","Roll","GreaterOrEqual","GlobalMaxPool","LessOrEqual","ResizeNearestNeighbor","ReduceMax","ReduceMin"
]

def operation_similarity(l1: str, l2: str):
    return difflib.SequenceMatcher(None, l1, l2).quick_ratio()

def auto_matching(onnx_model):
    model_copy = deepcopy(onnx_model)
    nonsupported_layer = []
    layer_count = 0
    for node_nonsupported in model_copy.graph.node:
        layer_count += 1
        if not (node_nonsupported.op_type in supported_list):
            print("matching op:", node_nonsupported.op_type)
            most_similar_op = None
            similarity = 0.
            for i in range(len(supported_list)):
                if operation_similarity(node_nonsupported.op_type, supported_list[i]) > similarity:
                    most_similar_op = supported_list[i]
            # input_name = node_nonsupported.input
            supported_op = onnx.helper.make_node(most_similar_op,
                                    name=node_nonsupported.name + '_matched',
                                    # inputs=[node_nonsupported.input[0]],
                                    inputs=node_nonsupported.input,
                                    # outputs=[node_nonsupported.output[0]]
                                    outputs=node_nonsupported.output
                                    )
            nonsupported_layer.append(node_nonsupported)
            # add_layers.append(LpNormalization)
            onnx_model.graph.node.insert(layer_count, supported_op)
            layer_count += 1
    for layer in nonsupported_layer:
        # print(layer.name)
        onnx_model.graph.node.remove(layer)

