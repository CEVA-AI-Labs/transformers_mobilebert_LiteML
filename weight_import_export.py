import argparse
import torch

from src.transformers.models.mobilebert.configuration_mobilebert import MobileBertConfig
from src.transformers.models.mobilebert.modeling_mobilebert import MobileBertForQuestionAnswering

from transformers.utils import logging
from mobilebert import MobileBERT
import tensorflow as tf
import numpy as np
import os
import json
import string
import re
from collections import Counter

def export_weights_to_numpy(path,out_folder):
    m = MobileBERT(path)
    tmp = m.interpreter.get_tensor_details()
    for block in tmp:
        name = block['name']
        val = m.interpreter.tensor(block['index'])
        if "/" in name:
            words = name.split("/")
            words = words[:-1]
            newPath = "/".join(words)
            newPath = os.path.join(out_folder,newPath)
            if not os.path.exists(newPath):
                os.makedirs(newPath)
        np.save(os.path.join(out_folder,name), val())

def load_torch_MobileBertForQuestionAnswering_model(json_conf_file_path,state_dict_file_path):
    config = MobileBertConfig.from_json_file(json_conf_file_path)
    model = MobileBertForQuestionAnswering(config)
    model.load_state_dict(torch.load(state_dict_file_path))
    model.eval()
    return model

def export_using_huggingface_to_pytorch(experiment_path):
    mobilebert_config_file = os.path.join(experiment_path,r"bert_config.json")
    tf_checkpoint_path = os.path.join(experiment_path,r"mobilebert_variables.ckpt")
    pytorch_dump_path = os.path.join(experiment_path,r"squad_pytorch_model.pt")
    # Initialise PyTorch model
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = MobileBertForQuestionAnswering(config)
    # Load weights from tf checkpoint
    model = transformers.load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)

def export_onnx_model(model,dim,output_path):
    model.transform_input = False
    model.eval().to(device='cuda')
    ids = torch.ones((1,dim),dtype=torch.int32)
    masks = torch.ones((1, dim), dtype=torch.int32)
    segments = torch.ones((1, dim), dtype=torch.int32)
    dummy_input = (ids.to(device='cuda'),masks.to(device='cuda'),segments.to(device='cuda'))
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.TRAINING,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def main():
    motherFolder = r"./mobilebert_float_384_20200602_weights/"
    tflitefile = r'mobilebert_float_384_20200602.tflite'
    onnx_path = r"../mobilebert_float_384_20200602_onnx/mobile_bert_model.onnx"
    #export_weights_to_numpy(tflitefile,motherFolder)   # cumbersome
    #tflite2onnx.convert(tflitefile, onnx_path)
    out_path = os.path.join(motherFolder,"uncased_torch.pt")

    experiment_path = r"./"
    #export_using_huggingface_to_pytorch(experiment_path)
    json_conf_file_path = os.path.join(experiment_path, r"bert_config.json")
    tf_checkpoint_path = os.path.join(experiment_path, r"mobilebert_variables.ckpt")
    state_dict_file_path = os.path.join(experiment_path, r"squad_pytorch_model.pt")
    output_path = os.path.join(experiment_path, r"squad_onnx_model.onnx")
    m = load_torch_MobileBertForQuestionAnswering_model(json_conf_file_path,state_dict_file_path)
    export_onnx_model(m, 384, output_path)

if __name__ == '__main__':
    main()