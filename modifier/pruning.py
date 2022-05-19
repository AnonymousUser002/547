from copy import deepcopy
from utils.utils import *

pruning_rule = [
    'DynamicQuantizeLinear',
    'DequantizeLinear',
    'Gemm',
]


def pruning(onnx_model):
    model_copy = deepcopy(onnx_model)
    redundant_layers = []
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    for node_pruning in model_copy.graph.node:
        if node_pruning.op_type == 'DequantizeLinear':
            if node_pruning.input[0] in weights:
                # layer_count += 1
                if len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 4:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 5:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 3:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 2:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 1:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).to(torch.float)

                for node in onnx_model.graph.node:
                    for i in range(len(node.input)):
                        if node.input[i] == node_pruning.output[0]:
                            # print('Find a quantized param:', node_input)
                            # print('intput name:', node.input[i])
                            node.input[i] = node.input[i] + "_modified"
                            dequant_tensor = onnx.helper.make_tensor(node.input[i],
                                                                    data_type=onnx.TensorProto.FLOAT,
                                                                    dims=dequant_weights.size(),
                                                                    vals=Torch2OnnxWeights(dequant_weights).raw_data,
                                                                    raw=True)
                            onnx_model.graph.initializer.append(dequant_tensor)
                # print(node_dequant.name, dequant_weights)
                redundant_layers.append(node_pruning)
        if node_pruning.op_type == 'DynamicQuantizeLinear':
            for node in onnx_model.graph.node:
                if node.op_type == 'DequantizeLinear' and node.input[0] == node_pruning.output[0]\
                     and node.input[1] == node_pruning.output[1]\
                     and node.input[2] == node_pruning.output[2]:
                    try:
                        weights[node_pruning.input[0]]
                    except:
                        for node_restore in onnx_model.graph.node:
                            for i in range(len(node_restore.input)):
                                if node.output[0] == node_restore.input[i]:
                                    node_restore.input[i] = node_pruning.input[0]
                    else:
                        for node_restore in onnx_model.graph.node:
                            for i in range(len(node_restore.input)):
                                if node.output[0] == node_restore.input[i]:
                                    restore_tensor = onnx.helper.make_tensor(node_restore.input[i],
                                                                            data_type=onnx.TensorProto.FLOAT,
                                                                            dims=OnnxWeights2Torch(weights[node_pruning.input[0]]).size(),
                                                                            # dims=[128],
                                                                            vals=weights[node_pruning.input[0]].raw_data,
                                                                            raw=True)
                                    onnx_model.graph.initializer.append(restore_tensor)
                                # else:
                                #     raise Exception("Ohhhhh Mingyi, type error for remove_DynamicQuant_layers!")
                    redundant_layers.append(node_pruning)
                    redundant_layers.append(node)
        if node_pruning.op_type == 'QuantizeLinear':
            if node_pruning.input[0] in weights:
                if len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 4:
                    quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                        / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1)\
                        + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)
        
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 5:
                    quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                        / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1, 1)\
                        + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 3:
                    quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                        / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1)\
                        + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 2:
                    quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                        / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1)\
                        + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 1:
                    quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                        / OnnxWeights2Torch(weights[node_pruning.input[1]])\
                        + OnnxWeights2Torch(weights[node_pruning.input[2]]).to(torch.float)).clip(0,255).to(torch.uint8)

                for node in onnx_model.graph.node:
                    for i in range(len(node.input)):
                        if node.input[i] == node_pruning.output[0]:
                            # print('Find a quantized param:', node_input)
                            node.input[i] = node.input[i] + "_modified"
                            # print(quant_weights.dtype)
                            # print(quant_weights.numpy().shape)
                            # print(OnnxWeights2Torch(Torch2OnnxWeights(quant_weights).raw_data).size())
                            quant_tensor = onnx.helper.make_tensor(node.input[i],
                                                                    data_type=onnx.TensorProto.UINT8,
                                                                    dims=quant_weights.size(),
                                                                    vals=Torch2OnnxWeights(quant_weights).raw_data,
                                                                    # vals=quant_weights.numpy(),
                                                                    raw=True
                                                                    )
                            onnx_model.graph.initializer.append(quant_tensor)
                redundant_layers.append(node_pruning)
        if node_pruning.op_type == 'Gemm':
            params = [weights[par_name] for par_name in node_pruning.input if par_name in weights]
            if len(params) == 3:
                kwargs = extract_attributes(node_pruning)
                # print(kwargs)
                if not kwargs["transpose_activation"]:
                    A = OnnxWeights2Torch(weights[node_pruning.input[0]])
                else:
                    A = OnnxWeights2Torch(weights[node_pruning.input[0]]).t()
                if not kwargs["transpose_weight"]:
                    B = OnnxWeights2Torch(weights[node_pruning.input[1]])
                else:
                    B = OnnxWeights2Torch(weights[node_pruning.input[1]]).t()
                C = OnnxWeights2Torch(weights[node_pruning.input[2]])
                actual_tensor = torch.nn.functional.linear(input=A, weight=B, bias=C)
                Gemm_add_tensor = onnx.helper.make_tensor(node_pruning.output[0],
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=actual_tensor.size(),
                                            # dims=[1],
                                            # vals=weights[node_quant.input[2]].raw_data,
                                            vals=Torch2OnnxWeights(actual_tensor).raw_data,
                                            raw=True
                                            )

                redundant_layers.append(node_pruning)
                onnx_model.graph.initializer.append(Gemm_add_tensor)
    for layer in redundant_layers:
        onnx_model.graph.node.remove(layer)