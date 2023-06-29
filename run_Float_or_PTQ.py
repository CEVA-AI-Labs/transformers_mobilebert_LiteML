import time
from time import perf_counter
import os
import sys, getopt
from enum import Enum
import torch
import onnx
import weight_import_export as wie
from mobilebert import MobileBERT
import tensorflow as tf
import numpy as np
import json
import string
import re
from collections import Counter
from ailabs_shared.load_config import load_config
from liteml.retrainer import RetrainerModel, RetrainerConfig

# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def parse_answers(json_dict):
    answers = {}
    for data in json_dict["data"]:
        for paragraph in data["paragraphs"]:
            for qa in paragraph['qas']:
                # new_answers = {qas["id"]: qas["answers"][0]["text"] for qas in paragraph["qas"]}
                # print(new_answers)
                # answers.update(new_answers)
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                new_answers = {qa["id"]: ground_truths}
                answers.update(new_answers)
    return answers


class qas:
    question: str
    answers: list


class jsondata:
    context: str
    qa: list


def remove_articles(text):
    # return re.sub(r'\b(the)\b', ' ', text)
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def remove_space(text):
    return text.replace(' ', '')


def trans_alph(text):
    text = re.sub('[åáāäạằ]', 'a', text)
    text = re.sub('[ćç]', 'c', text)
    text = re.sub('[đ]', 'd', text)
    text = re.sub('[éèěē]', 'e', text)
    text = re.sub('[íī]', 'i', text)
    text = re.sub('[ń]', 'n', text)
    text = re.sub('[óõöō]', 'o', text)
    text = re.sub('[Ś]', 's', text)
    text = re.sub('[ùüû]', 'u', text)
    return text


def lower(text):
    return text.lower()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_articles(remove_punc(trans_alph(lower(s)))))


def normalize_answer2(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return remove_space(white_space_fix(remove_articles(remove_punc(trans_alph(lower(s))))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # print(prediction, ' : ', ground_truth)
    # print(prediction_tokens, ' : ', ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score2(prediction, ground_truth, sw):
    if False == sw:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        prediction_tokens = normalize_answer2(prediction).split()
        ground_truth_tokens = normalize_answer2(ground_truth).split()

    # print(prediction, ' : ', ground_truth)
    # print(prediction_tokens, ' : ', ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # print(common)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_score2(prediction, ground_truth, sw):
    if False == sw:
        return (normalize_answer(prediction) == normalize_answer(ground_truth))
    else:
        return (normalize_answer2(prediction) == normalize_answer2(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, sw):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, sw)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def print_tensor_data(initializer: onnx.TensorProto) -> None:

    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        print(initializer.float_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        print(initializer.int32_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        print(initializer.int64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(initializer.double_data)
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        print(initializer.uint64_data)
    else:
        raise NotImplementedError

    return
class script_type(Enum):
    TFLite_Float = 1
    TFLite_8Bit = 2
    PyTorch_Float = 3
    PyTorch_8Bit = 4


def main(argv):
    # here we select the type of script we wish to run

    modes = ["TFLite_Float", "TFLite_8Bit", "PyTorch_Float", "PyTorch_8Bit"]
    devices = ["cpu", "gpu"]
    device = devices[0]

    if len(argv) == 1 and argv[0] == "help":
        print("____________________________________________________________________________________\n"
              "This is a script to run the MobileBert model on a test dataset\n"
              "There are 4 work modes:\n"
              "1) TFLite_Float\n"
              "Where we run the original TFLite floating point model\n"
              "2) TFLite_8Bit\n"
              "Where we run the original TFLite 8 bit fixed point model\n"
              "3) PyTorch_Float\n"
              "Where we run the converted PyTorch floating point model\n"
              "4) PyTorch_8Bit\n"
              "Where we run PTQ on the converted PyTorch floating point model\n\n"
              "The default device is cpu, but PyTorch_8Bit work mode supports gpu as well.\n"
              "Arguments should be passed as follows:\n"
              "run_Float_or_PTQ.py work_mode device\n"
              "where work_mode is one of:\n"
              "TFLite_Float, TFLite_8Bit, PyTorch_Float or PyTorch_8Bit\n"
              "and the device is optional (default=cpu) and may be cpu or gpu\n"
              "For example:\n"
              "run_Float_or_PTQ.py TFLite_Float\n"
              "or:\n"
              "run_Float_or_PTQ.py PyTorch_8Bit gpu\n"
              "____________________________________________________________________________________\n"
              )
        exit()
    elif len(argv) > 2:
        print("Cannot have more than 2 input arguments, " + str(len(argv)) + " were given:")
        print(str(argv))
        exit()
    else:
        if argv[0] not in modes:
            print(argv[0] + " is a wrong operation mode, options are:")
            for val in modes:
                print(val)
            exit()
        if len(argv) == 2:
            if argv[1] not in devices:
                print(argv[1] + " is a wrong device, options are:")
                for val in devices:
                    print(val)
                exit()
    if len(argv) == 2:
        if argv[1] in devices:
            if argv[1] == "gpu":
                device = "cuda"
            elif argv[1] == "cpu":
                device = "cpu"
    # here you choose if you wish to run the float version or the quantized 8 bit version of the give tflite model
    # even if we want to run only the torch model, we have to also load the tflite model since it has a lot of functionality we need

    current_script = eval("script_type." + argv[0])

    tflitefiles = ['../mobilebert_float_384_20200602.tflite', '../mobilebert_int8_384_20200602.tflite']
    tflitefile = tflitefiles[0]
    if argv[0] == "TFLite_8Bit":
        tflitefile = tflitefiles[1]

    m1 = MobileBERT(tflitefile)

    experiment_path = "./"
    json_conf_file_path = mobilebert_config_file = os.path.join(experiment_path, r"bert_config.json")
    state_dict_file_path = os.path.join(experiment_path, r"squad_pytorch_model.pt")
    m2 = wie.load_torch_MobileBertForQuestionAnswering_model(json_conf_file_path, state_dict_file_path)
    # image_dim = 384
    # dummy_input = np.zeros((1, image_dim))
    # dummy_input = torch.IntTensor(dummy_input)
    # m2(dummy_input,dummy_input,dummy_input)
    # torch.onnx.export(
    #    m2,
    #    (dummy_input,dummy_input,dummy_input),
    #    "MobileBertBeforeWrappingOpSet16_2SplitFix_pytorch_1_12_minFix_zeroFixFloatFixMaskFixConcatFixMinFixLatest3.ONNX",
    #    opset_version=16,
    #    do_constant_folding=True,
    #    training=torch.onnx.TrainingMode.EVAL,
    # )
    # exit()
    #model = onnx.load("MobileBertBeforeWrapping.ONNX")
    #model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    #onnx.checker.check_model(model)
    #graph_def = model.graph
    #outValueInfo = open("value_info","w")
    #outInitializer = open("initializer","w")
    #for v in graph_def.value_info:
    #    outValueInfo.write(str(v))
    #initializers = graph_def.initializer

    # Modify initializer
    # for initializer in initializers:
    #     # Data type:
    #     # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
    #     outInitializer.write("Tensor information:"+"\n")
    #     outInitializer.write(
    #         f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}"
    #     +"\n")
    #     outInitializer.write("Tensor value before modification:"+"\n")
    #     outInitializer.write(str(print_tensor_data(initializer))+"\n")

    expression = "m2." + device + "()"
    eval(expression)

    # 特殊記号を含む回答の場合、正しく単語を認識しないが、回答の結果としてはあっているものがあるため、その場合は、比較前のプレ処理にスペース削除のひと手間を加える。
    # 例 Levi's Stadiumが正解の場合、モデルの回答は levi ' s stadiumが返ってくる。正解値はlevi's stadium
    # 前処理として特殊記号を削除するため、 levi s studiamとなる。正解値はlevis stadium。
    # この場合、本来なら正解として算出していいが、F1値が下がってしまうため、最終的に全スペースを削除して1つの文字列として比較する。
    spdel = [
        [3, 13, 19, 20, 28, 41, 118, 122, 123, 124, 126, 131, 133, 159, 163, 168, 177, 223, 229, 233, 235, 291, 296,
         297, 301, 305, 309, 310, 328, 330, 336, 342, 423, 428, 562, 585, 588, 604, 605, 608, 609, 613, 615, 620, 664,
         743, 772, 781, 796, 833, 847, 852, 866, 867, 882, 907, 947, 969, 974, 976, 983, 1006, 1009, 1030, 1031, 1035,
         1036, 1037, 1038, 1050, 1054, 1057, 1071, 1076, 1118, 1136, 1164, 1211, 1212, 1240,
         1267, 1268, 1304, 1307, 1308, 1327, 1333, 1347, 1366, 1385, 1388, 1415, 1418, 1421, 1423, 1426, 1448, 1453,
         1461, 1462, 1507, 1510, 1519, 1531, 1532, 1544, 1558, 1560, 1579, 1629, 1680, 1749, 1761, 1770, 1773, 1789,
         1797, 1804, 1805, 1816, 1828, 1842, 1848, 1850, 1853, 1865, 1876, 1879, 1898, 1915, 1928, 1936, 1937, 1953,
         1967, 2011, 2027, 2031, 2046, 2052, 2073, 2084, 2086, 2087, 2109, 2112, 2122, 2123, 2138,
         2150, 2176, 2212, 2293, 2349, 2365, 2448, 2453, 2526, 2552, 2570, 2572, 2583, 2588, 2591, 2609, 2614, 2618,
         2643, 2676, 2680, 2682, 2727, 2750, 2752, 2756, 2765, 2766, 2772, 2776, 2797, 2837, 2850, 2860, 2880, 2892,
         2894, 2899, 2901, 2903, 2904, 2910, 2921, 2929, 2944, 2958, 2970, 2973, 2975, 2979, 2987, 2989, 2998, 3017,
         3019, 3035, 3037, 3038, 3043, 3066, 3067, 3102, 3117, 3119, 3120, 3139, 3160, 3162, 3192,
         3207, 3216, 3223, 3233, 3245, 3265, 3268, 3307, 3310, 3384, 3423, 3424, 3499, 3511, 3523, 3575, 3585, 3627,
         3632, 3641, 3653, 3669, 3680, 3683, 3684, 3705, 3745, 3752, 3769, 3779, 3802, 3844, 3910, 3941, 3942, 3948,
         3950, 3952, 3960, 3970, 3972, 3988, 3989, 3994, 4022, 4024, 4030, 4046, 4076, 4089, 4162, 4165, 4166, 4183,
         4208, 4209, 4340, 4415, 4478, 4485, 4493, 4495, 4509, 4510, 4514, 4524, 4534, 4538, 4545,
         4548, 4550, 4556, 4557, 4559, 4565, 4571, 4576, 4579, 4592, 4594, 4596, 4597, 4598, 4599, 4603, 4604, 4606,
         4608, 4611, 4623, 4625, 4636, 4663, 4666, 4669, 4681, 4683, 4692, 4708, 4753, 4805, 4807, 4816, 4842, 4845,
         4857, 4863, 4895, 4907, 4929, 4946, 4947, 4959, 4965, 4966, 4967, 4969, 4970, 4971, 4973, 4974, 4975, 4976,
         4977, 4978, 4979, 4981, 4983, 5120, 5124, 5161, 5212, 5217, 5250, 5324, 5343, 5354, 5355,
         5359, 5369, 5370, 5397, 5418, 5431, 5432, 5441, 5447, 5458, 5496, 5538, 5555, 5595, 5607, 5615, 5617, 5619,
         5660, 5706, 5708, 5719, 5725, 5729, 5736, 5742, 5762, 5774, 5798, 5799, 5800, 5816, 5818, 5830, 5856, 5862,
         5863, 5864, 5876, 5891, 5897, 5900, 5901, 5916, 5929, 5945, 5998, 6008, 6024, 6030, 6031, 6068, 6076, 6086,
         6096, 6103, 6113, 6117, 6119, 6125, 6137, 6154, 6158, 6161, 6168, 6181, 6187, 6198, 6202,
         6222, 6226, 6229, 6230, 6231, 6254, 6258, 6265, 6266, 6274, 6279, 6280, 6287, 6289, 6290, 6307, 6311, 6319,
         6320, 6346, 6349, 6368, 6403, 6409, 6417, 6420, 6453, 6512, 6553, 6604, 6609, 6628, 6630, 6690, 6696, 6722,
         6737, 6738, 6743, 6759, 6761, 6767, 6773, 6811, 6831, 6835, 6837, 6859, 6864, 6916, 7004, 7037, 7038, 7041,
         7042, 7051, 7072, 7150, 7158, 7159, 7173, 7192, 7215, 7217, 7220, 7224, 7237, 7238, 7239,
         7240, 7243, 7249, 7250, 7270, 7271, 7291, 7293, 7305, 7306, 7314, 7327, 7331, 7342, 7344, 7352, 7354, 7355,
         7369, 7406, 7407, 7408, 7414, 7417, 7419, 7430, 7431, 7459, 7463, 7467, 7471, 7513, 7518, 7534, 7535, 7538,
         7556, 7559, 7562, 7565, 7568, 7570, 7571, 7603, 7604, 7641, 7642, 7664, 7665, 7670, 7671, 7676, 7739, 7748,
         7791, 7795, 7801, 7803, 7836, 7865, 7877, 7880, 7916, 7923, 7956, 7959, 7976, 8072, 8080,
         8086, 8091, 8111, 8154, 8173, 8206, 8208, 8209, 8220, 8230, 8238, 8242, 8284, 8287, 8294, 8295, 8302, 8303,
         8304, 8305, 8308, 8316, 8329, 8367, 8400, 8436, 8486, 8502, 8506, 8553, 8567, 8579, 8628, 8637, 8641, 8651,
         8657, 8661, 8691, 8695, 8715, 8716, 8719, 8721, 8736, 8739, 8749, 8750, 8771, 8784, 8787, 8788, 8798, 8800,
         8801, 8810, 8822, 8846, 8855, 8863, 8871, 8874, 8879, 8883, 8894, 8899, 8908, 8936, 8965,
         8968, 9011, 9012, 9018, 9019, 9027, 9032, 9066, 9077, 9079, 9097, 9131, 9132, 9136, 9151, 9160, 9163, 9211,
         9212, 9225, 9226, 9228, 9231, 9232, 9239, 9248, 9250, 9256, 9259, 9269, 9271, 9278, 9303, 9308, 9315, 9317,
         9318, 9331, 9334, 9335, 9338, 9340, 9372, 9388, 9398, 9401, 9411, 9416, 9454, 9494, 9502, 9510, 9539, 9544,
         9569, 9577, 9595, 9614, 9615, 9617, 9620, 9623, 9626, 9627, 9628, 9634, 9673, 9680, 9683,
         9696, 9702, 9704, 9712, 9714, 9788, 9848, 9885, 9898, 9903, 9943, 9947, 9951, 9956, 9978, 9989, 9998, 10000,
         10015, 10044, 10064, 10072, 10102, 10110, 10127, 10130, 10132, 10136, 10150, 10153, 10176, 10196,
         10207, 10235, 10247, 10269, 10270, 10276, 10282, 10283, 10288, 10325, 10338, 10339, 10341, 10346, 10353, 10388,
         10411, 10413, 10418, 10434, 10446, 10448, 10450, 10451, 10452, 10456, 10458, 10483, 10490, 10499, 10514, 10535,
         10555, 10567, 10574, 10578, 10584, 10598, 10599, 10607, 10616, 10651, 10662, 10667, 10692, 10699, 10737, 10759,
         10795, 10799, 10809], [13, 19, 20, 41, 43, 118, 120, 122, 123, 124, 126, 131,
                                133, 159, 163, 168, 177, 223, 229, 233, 235, 291, 296, 297, 301, 305, 309, 310, 328,
                                336, 342, 345, 417, 423, 428, 551, 562, 585, 588, 604, 605, 608, 609, 613, 615, 620,
                                664, 743, 796, 797, 833, 847, 852, 860, 866, 867, 882, 907, 952, 969, 974, 983, 1006,
                                1009, 1030, 1031, 1035, 1036, 1037, 1050, 1054, 1071, 1076, 1118, 1136, 1164, 1178,
                                1190, 1211, 1240, 1267, 1304, 1307, 1327, 1333, 1347, 1385, 1391, 1421,
                                1422, 1423, 1426, 1446, 1448, 1450, 1453, 1461, 1462, 1507, 1510, 1519, 1531, 1544,
                                1558, 1560, 1629, 1680, 1761, 1770, 1773, 1776, 1789, 1804, 1805, 1816, 1828, 1837,
                                1839, 1848, 1850, 1865, 1876, 1898, 1928, 1937, 1953, 1967, 1986, 2011, 2027, 2031,
                                2046, 2052, 2073, 2084, 2086, 2087, 2108, 2109, 2112, 2122, 2123, 2138, 2150, 2157,
                                2176, 2212, 2235, 2293, 2349, 2352, 2365, 2448, 2453, 2475, 2526, 2546, 2547,
                                2552, 2570, 2583, 2588, 2591, 2609, 2614, 2618, 2643, 2645, 2676, 2680, 2682, 2750,
                                2752, 2765, 2766, 2772, 2776, 2797, 2837, 2850, 2880, 2892, 2894, 2899, 2901, 2903,
                                2904, 2910, 2921, 2929, 2944, 2958, 2960, 2970, 2973, 2975, 2977, 2979, 2985, 2987,
                                2989, 3017, 3019, 3035, 3038, 3043, 3066, 3102, 3117, 3119, 3120, 3139, 3160, 3162,
                                3192, 3194, 3207, 3216, 3223, 3233, 3245, 3265, 3268, 3307, 3310, 3384, 3423,
                                3424, 3469, 3499, 3511, 3523, 3575, 3578, 3585, 3627, 3632, 3641, 3653, 3669, 3670,
                                3680, 3683, 3684, 3769, 3779, 3782, 3802, 3821, 3844, 3910, 3934, 3941, 3942, 3948,
                                3950, 3952, 3960, 3970, 3972, 3988, 3989, 3994, 4022, 4024, 4030, 4038, 4046, 4076,
                                4089, 4162, 4165, 4166, 4208, 4209, 4340, 4415, 4438, 4478, 4485, 4495, 4510, 4514,
                                4524, 4545, 4550, 4556, 4557, 4559, 4565, 4571, 4592, 4594, 4596, 4597, 4599,
                                4606, 4608, 4609, 4611, 4625, 4629, 4636, 4643, 4663, 4666, 4681, 4683, 4692, 4708,
                                4753, 4805, 4807, 4816, 4842, 4857, 4863, 4895, 4907, 4929, 4946, 4947, 4959, 4965,
                                4966, 4967, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4981,
                                4983, 5078, 5085, 5120, 5124, 5161, 5162, 5212, 5217, 5226, 5250, 5324, 5354, 5355,
                                5369, 5380, 5388, 5397, 5400, 5418, 5431, 5432, 5441, 5447, 5458, 5496, 5538,
                                5555, 5595, 5607, 5615, 5617, 5619, 5660, 5706, 5708, 5725, 5729, 5736, 5742, 5762,
                                5774, 5799, 5800, 5816, 5818, 5830, 5854, 5856, 5863, 5864, 5876, 5891, 5901, 5916,
                                5929, 5945, 5998, 6008, 6024, 6030, 6031, 6068, 6076, 6086, 6103, 6113, 6117, 6119,
                                6125, 6158, 6161, 6168, 6181, 6187, 6198, 6202, 6222, 6226, 6229, 6230, 6231, 6254,
                                6258, 6265, 6266, 6274, 6279, 6280, 6284, 6287, 6289, 6290, 6307, 6311, 6319,
                                6320, 6346, 6349, 6368, 6403, 6409, 6417, 6420, 6512, 6553, 6581, 6604, 6609, 6628,
                                6652, 6690, 6696, 6737, 6739, 6743, 6759, 6761, 6773, 6811, 6823, 6831, 6835, 6837,
                                6859, 6864, 7004, 7037, 7038, 7041, 7042, 7051, 7072, 7135, 7158, 7159, 7173, 7190,
                                7192, 7215, 7217, 7220, 7224, 7237, 7238, 7239, 7240, 7243, 7249, 7250, 7271, 7291,
                                7293, 7305, 7306, 7314, 7331, 7342, 7343, 7344, 7352, 7354, 7355, 7369, 7406,
                                7408, 7414, 7417, 7419, 7430, 7431, 7459, 7463, 7467, 7471, 7495, 7513, 7518, 7534,
                                7535, 7538, 7556, 7559, 7562, 7563, 7568, 7570, 7603, 7641, 7642, 7664, 7670, 7671,
                                7676, 7748, 7791, 7801, 7803, 7877, 7880, 7916, 7923, 7956, 7959, 7976, 7979, 7980,
                                8072, 8080, 8111, 8154, 8173, 8206, 8208, 8220, 8230, 8238, 8242, 8284, 8287, 8294,
                                8295, 8302, 8303, 8304, 8305, 8308, 8316, 8329, 8367, 8400, 8436, 8486, 8502,
                                8506, 8553, 8567, 8628, 8637, 8641, 8651, 8657, 8661, 8691, 8695, 8715, 8719, 8749,
                                8771, 8784, 8787, 8788, 8798, 8800, 8822, 8846, 8855, 8863, 8871, 8874, 8883, 8894,
                                8908, 8926, 8965, 9002, 9011, 9012, 9018, 9019, 9027, 9032, 9077, 9079, 9091, 9092,
                                9131, 9132, 9136, 9137, 9151, 9160, 9163, 9211, 9212, 9219, 9225, 9226, 9228, 9231,
                                9232, 9239, 9248, 9250, 9256, 9259, 9269, 9271, 9278, 9295, 9303, 9308, 9315,
                                9318, 9331, 9335, 9338, 9340, 9352, 9372, 9388, 9398, 9401, 9416, 9454, 9494, 9510,
                                9539, 9544, 9569, 9577, 9581, 9595, 9614, 9615, 9617, 9620, 9626, 9627, 9628, 9634,
                                9673, 9680, 9683, 9702, 9704, 9712, 9714, 9788, 9848, 9885, 9898, 9903, 9947, 9951,
                                9956, 9978, 9989, 9998, 10000, 10015, 10064, 10099, 10110, 10127, 10130, 10132, 10136,
                                10150, 10176, 10196, 10207, 10235, 10247, 10269, 10270, 10276, 10282, 10283,
                                10325, 10338, 10339, 10341, 10346, 10353, 10388, 10418, 10434, 10446, 10448, 10450,
                                10452, 10456, 10458, 10483, 10490, 10499, 10514, 10515, 10535, 10555, 10574, 10578,
                                10584, 10598, 10599, 10607, 10616, 10651, 10662, 10667, 10699, 10759, 10795, 10799,
                                10809]]

    print('--------------------------------------------------------------------------------------')

    filename = "d.json"
    with open(filename, "r") as f:
        json_dict = json.load(f)
    answers = parse_answers(json_dict)

    pat = 0
    raw_dataset_test = tf.data.TFRecordDataset('squad_eval.tfrecord', "ZLIB")
    raw_dataset_calib = tf.data.TFRecordDataset('squad_eval.tfrecord', "ZLIB")
    example = tf.train.Example()
    tm_perf = []

    if current_script == script_type.PyTorch_8Bit:
        resultfile = 'result_' + tflitefile.replace('../', '') + '.txt'
    else:
        resultfile = 'result_' + str(current_script) + '.txt'
    f = open(resultfile, 'w')

    f1 = exact_match = total = 0
    pre_f1 = f1

    # 全項目数が10833問ある。お試しの場合は、項目数を減らして実施する。
    # 最終的な精度は全問実施しないと一致しないので注意

    #QUESTIONS = 10833  # this is the full dataset size
    QUESTIONS_CALIB = 10
    QUESTIONS_TEST = 10833
    if current_script == script_type.PyTorch_8Bit:
        calib_list = []
        for calib_sample in raw_dataset_calib.take(QUESTIONS_CALIB):
            example.ParseFromString(calib_sample.numpy())
            ent = example.features.feature['tokens']
            stokens = []
            for val in ent.bytes_list.value:
                stokens.append(val.decode(encoding='utf-8'))
            if len(stokens) > m1.max_length:
                raise IndexError("Token length more than max seq length!")
                print("Max exceeded")
            torch_ids = m1.get_ids(stokens)
            torch_masks = m1.get_masks(stokens)
            torch_segments = m1.get_segments(stokens)
            torch_ids = torch.IntTensor(torch_ids)
            torch_ids = eval("torch_ids." + device + "()")
            torch_masks = torch.IntTensor(torch_masks)
            torch_masks = eval("torch_masks." + device + "()")
            torch_segments = torch.IntTensor(torch_segments)
            torch_segments = eval("torch_segments." + device + "()")
            calib_list.append((torch_ids, torch_masks, torch_segments))
        calibration_loader = torch.utils.data.DataLoader(
            calib_list, batch_size=1, num_workers=0, )

        ptq_config = r"quantization_config_.yaml"
        rtrnrCfg = RetrainerConfig(ptq_config)
        rtrnrCfg.optimizations_config["QAT"]["calibration_loader"] = calibration_loader
        calibration_loader_key = lambda m, x: m(x[0], x[1], x[2])
        rtrnrCfg.optimizations_config["QAT"]["calibration_loader_key"] = calibration_loader_key
        m2 = RetrainerModel(m2, rtrnrCfg)
        m2.initialize_quantizers(calibration_loader,key = calibration_loader_key)
        eval(expression)
        m2.training = False
        def set_model_to_eval(model):
            for name,layer in model._modules.items():
                if hasattr(layer,"training"):
                    layer.training = False
                set_model_to_eval(layer)

        set_model_to_eval(m2)

    now = time.time()
    if current_script == script_type.PyTorch_8Bit and device=="cuda":

        # run the experiment fast on a GPU
        batch_size = 5
        print("running inference on GPU with batch size of " + str(batch_size))
        counter = 0
        batch = []
        list_of_tm = []
        for test_sample in raw_dataset_test.take(QUESTIONS_TEST):
            counter += 1
            batch.append(test_sample)
            if counter%batch_size ==0 or counter == QUESTIONS_TEST:
                counter=0
                # now run inference with GPU
                torch_ids_list = []
                torch_masks_list = []
                torch_segments_list = []
                stokens_list = []
                ans_list = []
                for test_sample_batch in batch:
                    stokens = []
                    example.ParseFromString(test_sample_batch.numpy())
                    ent = example.features.feature['qas_id']
                    for val in ent.bytes_list.value:
                        qas_id = val.decode(encoding='utf-8')
                    ans = answers[str(qas_id)]
                    ans_list.append(ans)

                    ent = example.features.feature['tokens']
                    for val in ent.bytes_list.value:
                        stokens.append(val.decode(encoding='utf-8'))
                    stokens_list.append(stokens)
                    torch_ids = m1.get_ids(stokens)
                    torch_ids = np.expand_dims(torch_ids, axis=0)
                    torch_masks = m1.get_masks(stokens)
                    torch_masks = np.expand_dims(torch_masks, axis=0)
                    torch_segments = m1.get_segments(stokens)
                    torch_segments = np.expand_dims(torch_segments, axis=0)

                    torch_ids = torch.IntTensor(torch_ids)
                    torch_ids = eval("torch_ids." + device + "()")
                    torch_ids_list.append(torch_ids)
                    torch_masks = torch.IntTensor(torch_masks)
                    torch_masks = eval("torch_masks." + device + "()")
                    torch_masks_list.append(torch_masks)
                    torch_segments = torch.IntTensor(torch_segments)
                    torch_segments = eval("torch_segments." + device + "()")
                    torch_segments_list.append(torch_segments)
                torch_ids = torch.cat(torch_ids_list,dim=0)
                torch_masks = torch.cat(torch_masks_list, dim=0)
                torch_segments = torch.cat(torch_segments_list,dim=0)
                st = perf_counter()
                res = m2(torch_ids, torch_masks, torch_segments)
                #print(str(res))
                tm = perf_counter() - st
                n = len(batch)
                list_of_tm.append(tm / n)
                batch=[]
                start_logits = np.asarray(res['start_logits'].cpu().detach())
                end_logits = np.asarray(res['end_logits'].cpu().detach())
                ends = tf.math.argmax(end_logits, output_type=tf.dtypes.int32, axis=1).numpy()
                starts = tf.math.argmax(start_logits, output_type=tf.dtypes.int32, axis=1).numpy()
                for i in range(n):
                    start = starts[i]
                    end = ends[i]
                    tokens = " ".join(stokens_list[i]).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")
                    answer = " ".join(stokens_list[i][start:end + 1]).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")
                    #print(answer)
                    total += 1
                    #print(total)
                    #print(ans_list[i])
                    val_f1, val_exact, tm_perf, exact_match, f1, pre_f1 = analyzer_results(tm_perf, tm, total,
                                                                                           spdel, pat, exact_match, f1,
                                                                                           pre_f1,
                                                                                           f, tokens, answer,ans_list[i])


    else:
        for test_sample in raw_dataset_test.take(QUESTIONS_TEST):
            total += 1
            #print(total)
            example.ParseFromString(test_sample.numpy())
            ent = example.features.feature['qas_id']
            for val in ent.bytes_list.value:
                qas_id = val.decode(encoding='utf-8')
            ans = answers[str(qas_id)]
            #print(ans)
            if current_script == script_type.TFLite_Float or current_script == script_type.TFLite_8Bit:
                tokens, answer, tm = m1.run4example(example)
            else:
                tokens, answer, tm = m1.run4example(example, m2)

            val_f1,val_exact,tm_perf, exact_match, f1, pre_f1 = analyzer_results(tm_perf,tm,total,spdel,pat,exact_match,f1,pre_f1,f,tokens,answer,ans)
    end = time.time()
    hours, rem = divmod(end - now, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print('total : ', total, ' exact_match : ', val_exact, ' f1_score : ', val_f1)
    print('total : ', total, ' exact_match : ', val_exact, ' f1_score : ', val_f1, file=f)

    print('min : ', min(tm_perf), 's\nmax : ', max(tm_perf), 's\navg : ', np.mean(tm_perf), 's\n')
    print('min : ', min(tm_perf), 's\nmax : ', max(tm_perf), 's\navg : ', np.mean(tm_perf), 's\n', file=f)

    pat += 1

    f.close()
def analyzer_results(tm_perf,tm,total,spdel,pat,exact_match,f1,pre_f1,f,tokens,answer,ans):
    tm_perf.append(tm)
    res = total in spdel[pat]
    #print('--- ', res)

    exact_match += metric_max_over_ground_truths(exact_match_score2, answer, ans, res)
    f1 += metric_max_over_ground_truths(f1_score2, answer, ans, res)

    val_exact = 100.0 * exact_match / total
    val_f1 = 100.0 * f1 / total

    #前回の回答時よりF1値が下がった(=間違えた)場合にのみ、画面、ファイルにその結果を出力
    if pre_f1 > val_f1:
        print(total, ' -----')
        print(total, ' -----', file=f)
        print('Q: ', tokens)  # 質問と前提文
        print('A: ', answer)  # モデルの回答結果
        print('Q: ', tokens, file=f)
        print('A: ', answer, file=f)
        print('C: ', ans)  # 正解値候補(複数)
        print('C: ', ans, file=f)
        print('total : ', total, ' exact_match : ', val_exact, ' f1_score : ', val_f1)
        print('total : ', total, ' exact_match : ', val_exact, ' f1_score : ', val_f1, file=f)

    pre_f1 = val_f1

    return val_f1,val_exact,tm_perf,exact_match,f1,pre_f1



if __name__ == '__main__':
    # This script runs the original model (float)
    main(sys.argv[1:])
