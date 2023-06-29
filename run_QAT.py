import re
import json
import time
from time import perf_counter
import os
import sys, getopt
from enum import Enum
import torch
import onnx
from typing import Iterable
from datetime import datetime
import weight_import_export
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
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
from transformers import DefaultDataCollator
from datasets import load_dataset
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
def take_out_nums_and_letters(string_input):
    newArr = []
    for vals in re.split('(\d+)', string_input):
        for val in re.split('([^a-zA-Z0-9])', vals):
            if val != " " and val != "":
                newArr.append(val.lower())
    return newArr
def preprocess_json(json_input,numOfSamples):
    counter = 0
    returning_list = []
    for title in json_input['data']:
        for paragraph in title['paragraphs']:
            for qas in paragraph['qas']:
                tmpArr = []
                tmpArr.append("[CLS]")
                for v in take_out_nums_and_letters(qas['question']):
                    tmpArr.append(v)
                tmpArr.append("[SEP]")
                for v in take_out_nums_and_letters(paragraph['context']):
                    tmpArr.append(v)
                tmpArr.append("[SEP]")
                returning_list.append(tmpArr)
                counter+=1
                if counter==numOfSamples:
                    return returning_list
    return returning_list
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

    set_device_expression = "m2." + device + "()"
    eval(set_device_expression)


    filename = "d.json"
    with open(filename, "r") as f:
        json_dict = json.load(f)
    answers = parse_answers(json_dict)

    #QUESTIONS = 10833  # this is the full dataset size
    QUESTIONS_CALIB = 10
    QUESTIONS_TEST = 10833
    pat = 0

    squad = load_dataset('squad',split="train[0:]")
    squad = squad.train_test_split(test_size=0.2)

    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    calib_list = convert_tokenized_to_proper_format(tokenized_squad['train'],QUESTIONS_CALIB,device,False)
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

    raw_dataset_test = tf.data.TFRecordDataset('squad_eval.tfrecord', "ZLIB")
    if current_script == script_type.PyTorch_8Bit:
        calibration_loader = torch.utils.data.DataLoader(
            calib_list, batch_size=1, num_workers=0, )

        ptq_config = r"quantization_config.yaml"
        rtrnrCfg = RetrainerConfig(ptq_config)
        rtrnrCfg.optimizations_config["QAT"]["calibration_loader"] = calibration_loader
        calibration_loader_key = lambda m, x: m(x[0], x[1], x[2])
        rtrnrCfg.optimizations_config["QAT"]["calibration_loader_key"] = calibration_loader_key
        m2.training=True
        eval(set_device_expression)


        m2 = RetrainerModel(m2, rtrnrCfg)
        m2.initialize_quantizers(calibration_loader, key=calibration_loader_key)

        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(
            output_dir="my_awesome_qa_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            push_to_hub=False,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=m2,
            args=training_args,
            train_dataset=tokenized_squad["train"],
            eval_dataset=tokenized_squad["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )



        trainer.train()


        def set_model_to_eval(model):
            for name,layer in model._modules.items():
                if hasattr(layer,"training"):
                    layer.training = False
                set_model_to_eval(layer)
        set_model_to_eval(m2)

    now = time.time()

    if current_script == script_type.PyTorch_8Bit and device=="cuda":

        # run the experiment fast on a GPU
        batch_size = 10
        print("running inference on GPU with batch size of " + str(batch_size))
        counter = 0
        batch = []
        list_of_tm = []

        counter = 0

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
                                                                                           exact_match, f1,
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

            val_f1,val_exact,tm_perf, exact_match, f1, pre_f1 = analyzer_results(tm_perf,tm,total,exact_match,f1,pre_f1,f,tokens,answer,ans)
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
def analyzer_results(tm_perf,tm,total,exact_match,f1,pre_f1,f,tokens,answer,ans):
    tm_perf.append(tm)
    res = False
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

def update_vocab(list_of_lists,current_dict,inverse_dict):
    for line in list_of_lists:
        for word in line:
            if word not in current_dict.keys():
                inverse_dict[len(inverse_dict)] = word
                current_dict[word] = len(current_dict)
    return


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def convert_tokenized_to_proper_format(dataset,num_of_samples,device,expand_dims):
    returned_list       = []
    input_ids_arr       = []
    attention_mask_arr  = []
    token_type_ids_arr  = []
    counter = 0
    for entry in dataset:
        input_ids           = torch.IntTensor(entry['input_ids'])
        if expand_dims:
            input_ids       = torch.unsqueeze(input_ids, axis=0)
        input_ids           = eval("input_ids." + device + "()")
        attention_mask      = torch.IntTensor(entry['attention_mask'])
        if expand_dims:
            attention_mask       = torch.unsqueeze(attention_mask, axis=0)
        attention_mask = eval("attention_mask." + device + "()")
        token_type_ids      = torch.IntTensor(entry['token_type_ids'])
        if expand_dims:
            token_type_ids       = torch.unsqueeze(token_type_ids, axis=0)
        token_type_ids = eval("token_type_ids." + device + "()")
        if expand_dims:
            input_ids_arr.append(input_ids)
            attention_mask_arr.append(attention_mask)
            token_type_ids_arr.append(token_type_ids)
        else:
            returned_list.append((input_ids,attention_mask,token_type_ids))
        counter+=1
        if counter==num_of_samples:
            if expand_dims:
                torch_ids = torch.cat(input_ids_arr, dim=0)
                torch_masks = torch.cat(attention_mask_arr, dim=0)
                torch_segments = torch.cat(token_type_ids_arr, dim=0)
                return torch_ids,torch_masks,torch_segments
            else:
                return returned_list

if __name__ == '__main__':
    # This script runs the original model (float)
    main(sys.argv[1:])
