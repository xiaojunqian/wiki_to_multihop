# -*- coding: utf-8 -*-
"""
多跳QA数据集生成器 (多进程修复版)
"""

import os
import json
import argparse
import re
import time
import logging
from multiprocessing import Pool
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
from openai import OpenAI

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 全局变量 (仅在子进程中有效) ---
WORKER_CLIENTS = []
WORKER_CONFIG = None
WORKER_PROMPT1 = ""
WORKER_PROMPT2 = ""
WORKER_GEN_CFG = {}

# --- 辅助工具函数 ---

def load_prompt_template(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt模板文件未找到: {filepath}")
        raise

def parse_llm_response_with_thinking(response_text: str) -> Tuple[str, str]:
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        actual_response_content = response_text[match.end():].strip()
        return thinking_content, actual_response_content
    return "", response_text.strip()

def parse_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    json_str = match.group(1) if match else text
    try:
        start_brace = json_str.find('{')
        end_brace = json_str.rfind('}')
        if start_brace != -1 and end_brace > start_brace:
            json_str = json_str[start_brace : end_brace + 1]
            json_str_cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str_cleaned)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None

# --- 子进程初始化与核心逻辑 ---

def worker_initializer(args, prompt1_text, prompt2_text):
    """
    子进程初始化函数：在这里创建OpenAI客户端，避免Pickle错误。
    """
    global WORKER_CLIENTS, WORKER_CONFIG, WORKER_PROMPT1, WORKER_PROMPT2, WORKER_GEN_CFG
    
    WORKER_CONFIG = args
    WORKER_PROMPT1 = prompt1_text
    WORKER_PROMPT2 = prompt2_text
    
    # 设置生成参数
    WORKER_GEN_CFG = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    
    # 初始化客户端列表
    WORKER_CLIENTS = []
    base_port = args.base_port
    host = args.host
    # 简单轮询创建
    for i in range(args.num_endpoints):
        try:
            client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{base_port + i}/v1")
            WORKER_CLIENTS.append(client)
        except Exception:
            pass

def _request_vllm(client: OpenAI, prompt: str) -> str:
    global WORKER_CONFIG, WORKER_GEN_CFG
    try:
        response = client.chat.completions.create(
            model=WORKER_CONFIG.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            **WORKER_GEN_CFG
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        # 新增：打印具体错误信息，方便调试
        print(f"API Error: {e}") 
        return ""

def _call_llm_step(client: OpenAI, prompt_template: str, **kwargs) -> Tuple[Optional[Dict], str, str]:
    formatted_prompt = prompt_template.format(**kwargs)
    raw_response = _request_vllm(client, formatted_prompt)
    if not raw_response:
        return None, "", "N/A (Request Failed)"
    
    thinking, response_content = parse_llm_response_with_thinking(raw_response)
    parsed_json = parse_and_repair_json(response_content)
    return parsed_json, thinking, raw_response

def process_single_item_task(args_tuple: Tuple[int, Dict]) -> Dict:
    """
    独立的任务处理函数，使用全局变量 WORKER_CLIENTS
    """
    global WORKER_CLIENTS, WORKER_PROMPT1, WORKER_PROMPT2
    
    original_idx, item = args_tuple
    if not WORKER_CLIENTS:
        return {"error": "No API clients available in worker."}

    # 轮询选择客户端
    client = WORKER_CLIENTS[original_idx % len(WORKER_CLIENTS)]
    debug_steps = []

    try:
        # 1. 解析数据
        r_list, b_list = [], []
        i = 1
        while f'document{i}' in item:
            r_list.append(item[f'document{i}']['text']) # 这里注意: nR_deal 输出的是 text 还是 R? 请根据实际情况调整
            # 注意：如果你使用的是经过 nR_deal 处理的数据，这里可能需要改为取 item[f'R{i}']
            # 为了兼容性，这里先保留取 text，实际使用建议检查数据结构
            
            conn_key = f'connection_d{i}_d{i+1}'
            if conn_key in item:
                b_list.append(item[conn_key]['bridge'])
            i += 1
        
        # 兼容 nR_deal 的输出 (如果有 R 字段则优先使用)
        final_r_list = []
        for idx in range(len(r_list)):
            r_key = f'R{idx+1}'
            if r_key in item and item[r_key]:
                 # item[r_key] 是个列表，取所有段落拼接，或者取第一段
                final_r_list.append(item[r_key][0])
            else:
                final_r_list.append(r_list[idx])
        # print(len(final_r_list[0]))
        # print(len(final_r_list[1]))
        # print(len(final_r_list[2]))
        # import pdb; pdb.set_trace()
        # return None
        if not final_r_list or len(final_r_list) != len(b_list) + 1:
             return {"error": "Invalid structure", "context": item}

        # 2. 初始化
        init_result, init_thinking, raw_resp1 = _call_llm_step(
            client, WORKER_PROMPT1, R_final=final_r_list[-1], B_final=b_list[-1]
        )
        debug_steps.append({"step": "initialization", "thinking": init_thinking, "raw_response": raw_resp1})
        
        if not init_result or "A" not in init_result or "Q_initial" not in init_result:
            return {"error": "Init failed", "debug_info": debug_steps, "context": item}

        answer = init_result['A']
        current_question = init_result['Q_initial']
        final_result = {"A": answer, "Q_chain": [{"level": 0, "question": current_question}]}

        # 3. 迭代
        for i in reversed(range(len(b_list))):
            level = len(b_list) - i
            replace_result, iter_thinking, raw_resp_iter = _call_llm_step(
                client,
                WORKER_PROMPT2,
                current_question=current_question,
                context_document=final_r_list[i],
                target_entity=b_list[i],
                anchor_entity=(b_list[i-1] if i > 0 else "")
            )
            debug_steps.append({"step": f"replacement_level_{level}", "thinking": iter_thinking, "raw_response": raw_resp_iter})

            if not replace_result or "Q_new" not in replace_result:
                return {"error": f"Step {level} failed", "debug_info": debug_steps, "context": item}

            current_question = replace_result['Q_new']
            final_result["Q_chain"].append({"level": level, "question": current_question})
        item["R1"] = [item["R1"][0]]
        item["R2"] = [item["R2"][0]]
        item["R3"] = [item["R3"][0]]
        final_result["context"] = item
        final_result["debug_info"] = debug_steps
        return final_result

    except Exception as e:
        return {"error": str(e), "context": item, "debug_info": debug_steps}

# --- 主流程 ---

class QAGenerator:
    def __init__(self, config):
        self.config = config
        # 不再这里初始化 clients

    def run(self):
        logger.info(f"Loading data from {self.config.input_file}...")
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f]
        except Exception as e:
            logger.error(f"Failed to load input: {e}")
            return

        start = self.config.start
        end = len(all_data) if self.config.end == -1 else self.config.end
        data_to_process = all_data[start:end]
        
        if not data_to_process:
            logger.warning("No data to process.")
            return
            
        process_args = list(enumerate(data_to_process, start=start))
        total_items = len(process_args)
        
        # worker_initializer(self.config, "", "")
        # for i in range(total_items):
        #     process_single_item_task(process_args[i])
        
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.failed_output_path), exist_ok=True)

        # 加载模板文本
        prompt1_content = load_prompt_template(self.config.prompt1_file)
        prompt2_content = load_prompt_template(self.config.prompt2_file)

        logger.info(f"Starting pool with {self.config.num_workers} workers...")
        
        # 使用 initializer 注入配置和客户端
        with Pool(processes=self.config.num_workers, 
                  initializer=worker_initializer, 
                  initargs=(self.config, prompt1_content, prompt2_content)) as pool:
            
            with open(self.config.output_path, "a", encoding='utf-8') as f_success, \
                 open(self.config.failed_output_path, "a", encoding='utf-8') as f_failed:
                
                # 使用独立的函数 process_single_item_task
                for result in tqdm(pool.imap_unordered(process_single_item_task, process_args), total=total_items):
                    if "error" in result:
                        f_failed.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        f_success.write(json.dumps(result, ensure_ascii=False) + "\n")
                        print("###""f{result}")
        logger.info("Done.")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--failed_output_path", type=str, required=True)
    parser.add_argument("--prompt1_file", type=str, required=True)
    parser.add_argument("--prompt2_file", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=8110)
    parser.add_argument("--num_endpoints", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--num_workers", type=int, default=64)
    return parser.parse_args()

if __name__ == "__main__":
    main = QAGenerator(parse_arguments())
    main.run()