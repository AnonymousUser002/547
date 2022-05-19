from copy import deepcopy
from utils.utils import *

translation_rule = [
    'QuantizeLinear',
    'DequantizeLinear',
    'SpaceToDepth',
    'DepthToSpace'
]

def translation(onnx_model):
    model_copy = deepcopy(onnx_model)
    translated_layers = []
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    layer_count = 0
    for node_trans in model_copy.graph.node:
        layer_count += 1
        if node_trans.op_type == 'DequantizeLinear':
            if node_trans.input[0] in weights:
                continue
            else:
                # print('dequant:', node_dequant.name)
                # layer_count += 1
                dequant_add = onnx.helper.make_node("Sub",
                                        name='dequantsub_'+node_trans.name,
                                        inputs=[node_trans.input[0], 'zeropoint_'+node_trans.name],
                                        outputs=['sub_result_'+node_trans.name])
                # print(OnnxWeights2Torch(weights[node_dequant.input[2]]).item())
                # todo: check the size not match in make tensor
                dequant_add_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # vals=weights[node_dequant.input[2]].raw_data,
                                        vals=Torch2OnnxWeights(OnnxWeights2Torch(weights[node_trans.input[2]]).to(torch.float)).raw_data,
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_add)
                # print(layer_count)
                # add_layers.append(dequant_add)
                onnx_model.graph.initializer.append(dequant_add_tensor)
                layer_count += 1
                dequant_mul = onnx.helper.make_node("Mul",
                                        name='dequantmul_'+node_trans.name,
                                        inputs=['sub_result_'+node_trans.name, 'scale_'+node_trans.name],
                                        outputs=[node_trans.output[0]])
                dequant_mul_tensor = onnx.helper.make_tensor('scale_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[1]]).size(),
                                        vals=weights[node_trans.input[1]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_dequant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_mul)
                # print(layer_count)
                # add_layers.append(dequant_mul)
                onnx_model.graph.initializer.append(dequant_mul_tensor)
                layer_count += 1
            translated_layers.append(node_trans)
        elif node_trans.op_type == 'QuantizeLinear':
            if node_trans.input[0] in weights:
                continue
            else:
                dequant_div = onnx.helper.make_node("Div",
                                        name='quantdiv_'+node_trans.name,
                                        inputs=[node_trans.input[0], 'scale_'+node_trans.name],
                                        outputs=['div_result_'+node_trans.name])
                dequant_mul_tensor = onnx.helper.make_tensor('scale_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[1]]).size(),
                                        vals=weights[node_trans.input[1]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_quant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_div)
                # add_layers.append(dequant_div)
                onnx_model.graph.initializer.append(dequant_mul_tensor)
                layer_count += 1
                dequant_add = onnx.helper.make_node("Add",
                                        name='quantadd_'+node_trans.name,
                                        inputs=['div_result_'+node_trans.name, 'zeropoint_'+node_trans.name],
                                        outputs=[node_trans.output[0]])

                dequant_add_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.UINT8,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # dims=[1],
                                        vals=weights[node_trans.input[2]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_quant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_add)
                # add_layers.append(dequant_add)
                onnx_model.graph.initializer.append(dequant_add_tensor)
                layer_count += 1
            translated_layers.append(node_trans)
    for layer in translated_layers:
        # print(layer.op_type)
        onnx_model.graph.node.remove(layer)