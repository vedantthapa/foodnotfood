#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   onnx.py
@Time    :   2021/12/11 11:25:34
@Author  :   Vedant Thapa 
@Contact :   thapavedant@gmail.com
'''

import torch
import warnings

warnings.filterwarnings(action='ignore')


def convert_model(torch_path, onnx_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(torch_path)
    if model.training:
        model.eval()
    dummy_input = torch.zeros(1, 3, 224, 224).to(device)
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11)
    print(f'Done. Model saved to {onnx_path}')


if __name__ == '__main__':
    torch_path = 'assets/model/efficientnet-B0-v1.pth'
    onnx_path = 'assets/model/efficientnet-B0-v1.onnx'
    convert_model(torch_path, onnx_path)
