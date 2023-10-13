# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/3/21 11:01
"""
    文件说明：
            
"""
import time

import torch
import os
import random
import numpy as np
import argparse
import logging
from model import PromptModel
from transformers import BertTokenizer
from data_set import PromptDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def ptunning(model,device,train_data,test_data,args,tokenizer):
    pass

def train(model, device, train_data, test_data, args, tokenizer):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        train_data: 训练数据类
        test_data: 测试数据类
        args: 训练参数配置信息
        tokenizer: 分词器
    Returns:
    """
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    # 构造训练所需的data_loader
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in args.requires_grad_params)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in args.requires_grad_params)], 'weight_decay': 0.0}
    ]

    # 冻结不训练的参数
    for name, param in model.named_parameters():
        if not any(r_name in name for r_name in args.requires_grad_params):
            param.requires_grad = False

    # 验证是否冻结成功
    requires_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            requires_grad_params.append(name)
            print("需要训练参数为{}，大小为{}".format(name, param.size()))
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.train()
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0

    words_ids = train_data.words_ids.to(device)
    words_ids_mask = train_data.words_ids_mask.to(device)
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            # 获取模型训练每个批次所需的输入内容，并放到对应设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_index = batch["mask_index"].to(device)
            label = batch["label"].to(device)

            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index,
                                    token_handler=args.token_handler,
                                    words_ids=words_ids, words_ids_mask=words_ids_mask,
                                    label=label)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                        (args.logging_steps * args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss

        # 每个Epoch对模型进行一次测试，记录测试集的损失
        eval_loss, eval_acc = evaluate(model, device, test_data, args)
        tb_write.add_scalar("test_loss", eval_loss, global_step)
        tb_write.add_scalar("test_acc", eval_acc, global_step)
        print("test_loss: {}, test_acc:{}".format(eval_loss, eval_acc))
        model.train()
        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()


def evaluate(model, device, test_data, args):
    """
    对测试数据集进行模型测试
    Args:
        model: 模型
        device: 设备信息
        test_data: 测试数据类
        args: 训练参数配置信息
    Returns:
    """
    # 构造测试集的DataLoader
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    total_loss, total = 0.0, 0.0
    words_ids = test_data.words_ids.to(device)
    words_ids_mask = test_data.words_ids_mask.to(device)

    y_true = []
    y_pre = []
    # 进行测试
    for step, batch in enumerate(iter_bar):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_index = batch["mask_index"].to(device)
            label = batch["label"].to(device)

            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index,
                                    token_handler=args.token_handler,
                                    words_ids=words_ids, words_ids_mask=words_ids_mask,
                                    label=label)
            loss = outputs[0]
            loss = loss.item()
            # 对loss进行累加
            total_loss += loss * len(batch["input_ids"])
            total += len(batch["input_ids"])
            # 记录原始标签和预测标签
            y_true.extend(batch["label"].numpy().tolist())
            y_pre.extend(outputs[2].cpu().numpy().tolist())

    # 计算最终测试集的loss和acc结果
    test_loss = total_loss / total
    test_acc = np.mean(np.array(y_true) == np.array(y_pre))
    return test_loss, test_acc


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()

    default_model_path = r"C:\Users\Administrator\Desktop\研究生相关\项目-问题文本分类\Bert-Pytorch-Chinese-TextClassification-master\bert\bert-tiny-chinese-4L"

    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--train_file_path', default='data/train.json', type=str, help='训练数据')
    parser.add_argument('--test_file_path', default='data/test.json', type=str, help='测试数据')
    parser.add_argument('--pretrained_model_path', default=default_model_path, type=str, help='预训练的BERT模型的路径')
    parser.add_argument('--data_dir', default='data/', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=32, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=16, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=10, type=int, help='保存训练日志的步数')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_len', type=int, default=256, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--token_handler', type=str, default="mean", help='答案映射标签多token策略')
    parser.add_argument('--template', type=str, default="{mask}满意。{text}", help='prompt模板')
    parser.add_argument('--pos_words', type=list, default=["很", "非常","比较","挺"], help='答案映射正标签对应标签词')
    parser.add_argument('--neg_words', type=list, default=["不","不太"], help='答案映射负标签对应标签词')
    parser.add_argument('--requires_grad_params', type=list, default=["cls.predictions"], help='模型训练参数')
    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 实例化PromptModel模型
    model = PromptModel.from_pretrained(args.pretrained_model_path)

    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=True)

    # 创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 根据正负标签的标签词构建答案空间映射字典
    label_dict = {"负向": {"label_words": args.neg_words, "label_id": 0},
                  "正向": {"label_words": args.pos_words, "label_id": 1}}
    # 加载训练数据和测试数据
    train_data = PromptDataSet(tokenizer, args.max_len, args.template, label_dict, args.data_dir, "train",
                               args.train_file_path)
    test_data = PromptDataSet(tokenizer, args.max_len, args.template, label_dict, args.data_dir, "test",
                              args.test_file_path)
    # 开始训练
    train(model, device, train_data, test_data, args, tokenizer)


if __name__ == '__main__':
    main()
