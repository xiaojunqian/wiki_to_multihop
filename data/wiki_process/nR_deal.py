# -*- coding: utf-8 -*-
"""
知识路径证据提炼工具

本脚本作为数据处理的中间步骤，用于从包含完整文章的多跳知识路径中，
为每个节点提取出最相关的、符合长度限制的文本片段（证据）。

工作流程:
1. 读取一个包含多跳路径的 JSONL 文件。
2. 对每条路径，自动检测其跳数。
3. 为路径中的每个文档（节点），根据其在路径中的位置（起点、中间、终点），
   提取与连接它的“桥梁实体”相关的段落。
4. 过滤掉所有超过预设Token长度的段落。
5. 如果路径中的任何一个节点都无法提取到有效的证据，则丢弃整条路径。
6. 将提炼后的路径（在原数据上增加 R1, R2, ... 字段）写入新的 JSONL 文件。
"""
import json
import re
import argparse
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 核心辅助函数 ---

def count_tokens(text: str) -> int:
    """通过按空格分割来简单估算文本的token数量。"""
    return len(text.split()) if isinstance(text, str) else 0

def split_into_paragraphs(text: str) -> List[str]:
    """使用正则表达式将文本分割成段落，并清理每个段落。"""
    if not text:
        return []
    # 使用正向后行断言在句子结束符和换行符后分割
    paragraphs = re.split(r'(?<=[.!?])\s*\n+', text.strip())
    # 清理每个段落，移除内部换行符并去除首尾空格
    return [p.strip().replace('\n', ' ') for p in paragraphs if p.strip()]

def extract_single_bridge_paragraphs(doc_text: str, bridge: str) -> List[str]:
    """从文档中提取所有包含单个指定桥梁实体的段落。"""
    if not (doc_text and bridge):
        return []
    paragraphs = split_into_paragraphs(doc_text)
    bridge_lower = bridge.lower()
    return [p for p in paragraphs if bridge_lower in p.lower()]

def extract_connecting_paragraphs(doc_text: str, bridge_in: str, bridge_out: str) -> List[str]:
    """
    为中间文档提取连接“入桥”和“出桥”的段落。

    策略:
    1. 优先寻找同时包含两个桥梁实体的段落。
    2. 如果找不到，则寻找并合并相邻的、分别包含两个桥梁的段落。
    """
    if not (doc_text and bridge_in and bridge_out and bridge_in != bridge_out):
        return []
        
    paragraphs = split_into_paragraphs(doc_text)
    bridge_in_lower = bridge_in.lower()
    bridge_out_lower = bridge_out.lower()
    
    # 策略1: 寻找同时包含两个bridge的段落
    paragraphs_with_both = [
        p for p in paragraphs if bridge_in_lower in p.lower() and bridge_out_lower in p.lower()
    ]
    if paragraphs_with_both:
        return paragraphs_with_both

    # 策略2: 合并分别包含两个bridge的相邻段落
    indices_in = {i for i, p in enumerate(paragraphs) if bridge_in_lower in p.lower()}
    indices_out = {i for i, p in enumerate(paragraphs) if bridge_out_lower in p.lower()}
    
    combined_snippets = set()
    for i_in in indices_in:
        for i_out in indices_out:
            # 检查相邻性
            if abs(i_in - i_out) == 1:
                start_idx, end_idx = sorted((i_in, i_out))
                combined_text = ' '.join(paragraphs[start_idx : end_idx + 1])
                combined_snippets.add(combined_text)

    return list(combined_snippets)

def refine_knowledge_path(path_data: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """
    处理单条知识路径，提取所有相关的证据片段 (R)。

    Args:
        path_data (Dict[str, Any]): 单条路径的原始JSON数据。
        max_tokens (int): 每个证据片段的最大token数。

    Returns:
        Dict[str, Any]: 如果路径有效，则返回增加了 R 字段的JSON数据；否则返回空字典。
    """
    # 动态检测跳数
    num_hops = 0
    while f'connection_d{num_hops+1}_d{num_hops+2}' in path_data:
        num_hops += 1
    if num_hops == 0:
        return {}

    all_references = {}
    
    for i in range(num_hops + 1):
        doc_idx = i + 1
        doc_data = path_data.get(f'document{doc_idx}', {})
        doc_text = doc_data.get('text', '')
        r_key = f'R{doc_idx}'
        
        extracted_paragraphs = []
        if i == 0:  # 首节点
            bridge_out = path_data.get('connection_d1_d2', {}).get('bridge')
            extracted_paragraphs = extract_single_bridge_paragraphs(doc_text, bridge_out)
            # 为首节点的段落加上标题前缀
            title = doc_data.get('title', '')
            if title:
                extracted_paragraphs = [f'The title is: {title}. {p}' for p in extracted_paragraphs]

        elif i == num_hops:  # 尾节点
            bridge_in = path_data.get(f'connection_d{num_hops}_d{num_hops+1}', {}).get('bridge')
            extracted_paragraphs = extract_single_bridge_paragraphs(doc_text, bridge_in)
            
        else:  # 中间节点
            bridge_in = path_data.get(f'connection_d{i}_d{i+1}', {}).get('bridge')
            bridge_out = path_data.get(f'connection_d{i+1}_d{i+2}', {}).get('bridge')
            extracted_paragraphs = extract_connecting_paragraphs(doc_text, bridge_in, bridge_out)
        
        # 过滤掉超长的段落
        filtered_paragraphs = [p for p in extracted_paragraphs if count_tokens(p) <= max_tokens]
        
        if not filtered_paragraphs:
            return {}  # 如果任何一步没有找到有效证据，则路径无效
            
        all_references[r_key] = filtered_paragraphs

    # 如果路径有效，则将提取的R字段添加到原始数据中
    refined_data = path_data.copy()
    for key, value in all_references.items():
        refined_data[key] = sorted(value, key=len) # 按长度排序
    return refined_data

# --- 主执行逻辑 ---

def main(args):
    """主函数，负责读取、处理和写入文件。"""
    logging.info(f"开始处理输入文件: {args.input_file}")
    
    valid_paths = []
    total_lines = 0
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            
            for line in tqdm(lines, desc="提炼证据路径"):
                try:
                    data = json.loads(line)
                    refined_data = refine_knowledge_path(data, args.max_tokens_per_snippet)
                    if refined_data:
                        valid_paths.append(refined_data)
                except json.JSONDecodeError:
                    logging.warning(f"跳过一个无法解析的JSON行。")
                    continue

        logging.info(f"处理完成。共处理 {total_lines} 行，其中 {len(valid_paths)} 条为有效路径。")

        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for item in valid_paths:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"结果已成功保存至: {args.output_file}")

    except FileNotFoundError:
        logging.error(f"错误: 输入文件未找到 -> {args.input_file}")
    except Exception as e:
        logging.error(f"处理过程中发生未知错误: {e}", exc_info=True)

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从知识路径中提炼相关的、符合长度限制的证据片段。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file", 
        type=str, 
        required=True, 
        help="输入的JSONL文件路径，包含原始的多跳知识路径。"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        required=True, 
        help="输出的JSONL文件路径，将包含提炼后的路径数据。"
    )
    parser.add_argument(
        "--max-tokens-per-snippet", 
        type=int, 
        default=1024, 
        help="每个提取出的证据片段允许的最大Token数（按空格估算）。"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
