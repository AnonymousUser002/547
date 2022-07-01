import os
import argparse
import torch
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx_pytorch.onnx_pytorch import ConvertModel
from modifier.pruning import pruning
from modifier.translation import translation
from modifier.auto_matching import auto_matching
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='path of tflite model', default='mobilenet_v1_1.0_224_quant')
parser.add_argument('--alpha', type=int, default=0.0, help='threshold for auto-matching')
parser.add_argument('--save_onnx', type=bool, default=False)
opt = parser.parse_args()
print(opt)

def onnx_modifier(onnx_model):
    pruning(onnx_model)
    translation(onnx_model)
    auto_matching(onnx_model, similarity=opt.alpha)
tflite_model_path = './tflite_model/'
model_path = tflite_model_path + opt.model_name + '.tflite'
inputs = generate_random_data(model_path)
tflite_out, output_details = test_tflite_results(model_path, inputs)
if opt.save_onnx == True:
    TfliteToOnnx(tflite_model_path)
onnx_model = onnx.load('./out_model/'+opt.model_name+'.onnx')
onnx_modifier(onnx_model)
onnx.save(onnx_model, './out_model/'+opt.model_name+'_modified.onnx')
onnx.checker.check_model(onnx_model)
pytorch_model = ConvertModel(onnx_model).eval()
output_pt = pytorch_model(torch.from_numpy(inputs[0]))
error = np.absolute(output_pt[output_details[0]['name']].detach().squeeze().to(torch.float).numpy()-tflite_out.squeeze())
print('final_error mean:', error.mean())
print('final_error_range:', (tflite_out.max() - tflite_out.min()))
print('final_error_normlized:', error.mean() / (tflite_out.max() - tflite_out.min() + 1.0e-08))
