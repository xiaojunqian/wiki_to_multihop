# -*- coding: utf-8 -*-
"""
多跳QA数据集生成器 (Prompt分离 + 防复读版 + 数据一致性修正)
功能：
1. 推理时使用完整文档片段或拼接片段 (final_r_list)。
2. System/User Prompt 分离，解决 Few-Shot 过拟合。
3. 防复读校验，拦截模型抄袭示例的行为。
4. 存储一致性修正：使用 Source Tracking 确保最终保存的 context 与喂给模型的输入完全一致。
"""

import os
import json
import argparse
import re
import time
import logging
import copy
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
    # 1. 提取代码块
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    json_str = match.group(1) if match else text
    
    # 2. 修复双花括号
    json_str = json_str.replace("{{", "{").replace("}}", "}")

    try:
        # 3. 重新定位最外层括号
        start_brace = json_str.find('{')
        end_brace = json_str.rfind('}')
        
        if start_brace != -1 and end_brace > start_brace:
            json_str = json_str[start_brace : end_brace + 1]
            # 4. 修复尾部逗号
            json_str_cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str_cleaned)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None

def get_smart_snippet(text: str, keywords: List[str], window: int = 4096) -> str:
    """
    智能截取函数：
    现在主要用于生成报错时的 Context 预览 (fallback)，
    或者当不需要严格一致性时的压缩手段。
    """
    if not text or not isinstance(text, str):
        return ""
    found_idx = -1
    target_kw = ""
    sorted_keywords = sorted([k for k in keywords if k], key=len, reverse=True)
    for kw in sorted_keywords:
        if kw in text:
            found_idx = text.index(kw)
            target_kw = kw
            break
    if found_idx == -1:
        if len(text) <= window * 2:
            return text
        return text[:window * 2] + "...(truncated_head)"
    start = max(0, found_idx - window)
    end = min(len(text), found_idx + len(target_kw) + window)
    snippet = text[start:end]
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"

def create_lean_context(item: Dict, keywords: List[str]) -> Dict:
    """
    仅用于生成错误日志中的 Context，不做逻辑上的严格要求。
    """
    lean_context = copy.deepcopy(item)
    try:
        keys_to_check = list(lean_context.keys())
        for key in keys_to_check:
            if key.startswith('R') and isinstance(lean_context[key], list):
                if len(lean_context[key]) > 0:
                    full_text = lean_context[key][0]
                    lean_context[key] = [get_smart_snippet(full_text, keywords)]
            elif key.startswith('document') and isinstance(lean_context[key], dict):
                if 'text' in lean_context[key]:
                    full_text = lean_context[key]['text']
                    lean_context[key]['text'] = get_smart_snippet(full_text, keywords)
    except Exception:
        pass
    return lean_context

# --- 子进程初始化与核心逻辑 ---

def worker_initializer(args, prompt1_text, prompt2_text):
    global WORKER_CLIENTS, WORKER_CONFIG, WORKER_PROMPT1, WORKER_PROMPT2, WORKER_GEN_CFG
    
    WORKER_CONFIG = args
    WORKER_PROMPT1 = prompt1_text
    WORKER_PROMPT2 = prompt2_text
    
    WORKER_GEN_CFG = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    
    WORKER_CLIENTS = []
    base_port = args.base_port
    host = args.host
    for i in range(args.num_endpoints):
        try:
            client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{base_port + i}/v1")
            WORKER_CLIENTS.append(client)
        except Exception:
            pass

def _request_vllm(client: OpenAI, messages: List[Dict]) -> str:
    global WORKER_CONFIG, WORKER_GEN_CFG
    try:
        response = client.chat.completions.create(
            model=WORKER_CONFIG.model_name,
            messages=messages,
            **WORKER_GEN_CFG
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"API Error: {e}") 
        return ""

def _call_llm_step(client: OpenAI, prompt_template: str, **kwargs) -> Tuple[Optional[Dict], str, str]:
    split_marker = "|||SPLIT|||"
    if split_marker in prompt_template:
        parts = prompt_template.split(split_marker)
        sys_prompt = parts[0].strip()
        user_prompt_template = parts[1].strip()
        user_prompt = user_prompt_template.format(**kwargs)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        formatted_prompt = prompt_template.format(**kwargs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ]

    raw_response = _request_vllm(client, messages)
    if not raw_response:
        return None, "", "N/A (Request Failed)"
    
    thinking, response_content = parse_llm_response_with_thinking(raw_response)
    parsed_json = parse_and_repair_json(response_content)
    return parsed_json, thinking, raw_response

def process_single_item_task(args_tuple: Tuple[int, Dict]) -> Dict:
    global WORKER_CLIENTS, WORKER_PROMPT1, WORKER_PROMPT2
    
    original_idx, item = args_tuple
    
    # 步骤 1: 提取关键词
    keywords = []
    if 'A' in item: keywords.append(item['A'])
    i = 1
    while True:
        conn_key = f'connection_d{i}_d{i+1}'
        if conn_key in item and 'bridge' in item[conn_key]:
            keywords.append(item[conn_key]['bridge'])
        else:
            break
        i += 1
        if i > 10: break

    # 步骤 2: 准备数据 (拼接 + 截断逻辑 + Source Tracking)
    final_r_list = []
    r_list_raw = []
    
    # [新增] 来源追踪：记录每一跳用的是什么类型 (R 或 Doc) 以及具体的内容
    source_tracking = [] 

    i = 1
    while f'document{i}' in item:
        r_list_raw.append(item[f'document{i}']['text'])
        i += 1
    
    for idx in range(len(r_list_raw)):
        doc_id = idx + 1
        r_key = f'R{doc_id}'
        doc_key = f'document{doc_id}'
        
        # 核心判断逻辑：只在这里写一次
        if r_key in item and item[r_key] and isinstance(item[r_key], list):
            try:
                # 使用 R (Retrieval Chunks)
                if len(item[r_key]) == 1:
                    content = item[r_key][0]
                else:
                    # 拼接逻辑
                    temp_i = 1
                    tempstr = item[r_key][0]
                    while(temp_i < len(item[r_key]) and len(tempstr) < 200):
                        tempstr += item[r_key][temp_i]
                        temp_i += 1
                    content = tempstr
                
                if "may refer to:" in content:
                    return {"error": "may refer to in content"}
                final_r_list.append(content)
                source_tracking.append({"type": "R", "key": r_key, "content": content})
                
            except Exception as e:
                return {"error": str(e)}
        else:
            # 使用 Document (Full Text)
            content = r_list_raw[idx]
            final_r_list.append(content)
            source_tracking.append({"type": "doc", "key": doc_key, "content": content})

    b_list = []
    i = 1
    while f'connection_d{i}_d{i+1}' in item:
        b_list.append(item[f'connection_d{i}_d{i+1}']['bridge'])
        i += 1

    # 仅用于报错时的 Context 预览
    lean_context_for_error = create_lean_context(item, keywords)

    if not final_r_list or len(final_r_list) != len(b_list) + 1:
         return {"error": "Invalid structure", "context": lean_context_for_error}

    if not WORKER_CLIENTS:
        return {"error": "No API clients available.", "context": lean_context_for_error}

    client = WORKER_CLIENTS[original_idx % len(WORKER_CLIENTS)]
    MAX_DEBUG_LEN = 20000
    debug_steps = []

    BANNED_PHRASES = [
        "Bolivarian Games", "Central American and Caribbean Games", "波多黎各",
        "BowTie Inc", "Dog World", "Robert Knight", "Knight Street",
        "Starochęciny", "Gmina Chęciny"
    ]

    try:
        # 步骤 3: LLM 初始化
        init_result, init_thinking, raw_resp1 = _call_llm_step(
            client, WORKER_PROMPT1, R_final=final_r_list[-1], B_final=b_list[-1]
        )
        
        debug_steps.append({
            "step": "initialization", 
            "thinking": init_thinking[:MAX_DEBUG_LEN] + "...",
            "raw_response": raw_resp1[:MAX_DEBUG_LEN] + "..." 
        })
        
        if not init_result or "A" not in init_result or "Q_initial" not in init_result:
            return {"error": "Init failed", "debug_info": debug_steps, "context": lean_context_for_error}

        answer = init_result['A']
        current_question = init_result['Q_initial']
        
        for phrase in BANNED_PHRASES:
            if phrase in current_question and phrase not in str(item):
                return {"error": f"Hallucinated example content: {phrase}", "debug_info": debug_steps, "context": lean_context_for_error}

        final_result = {"A": answer, "Q_chain": [{"level": 0, "question": current_question}]}

        # 步骤 4: LLM 迭代生成
        for i in reversed(range(len(b_list))):
            level = len(b_list) - i
            replace_result, iter_thinking, raw_resp_iter = _call_llm_step(
                client,
                WORKER_PROMPT2,
                current_question=current_question,
                last_document=final_r_list[i+1],
                context_document=final_r_list[i],
                target_entity=b_list[i],
                occupy_entity=(b_list[i-1] if i > 0 else "")
                # anchor_entity=(b_list[i-1] if i > 0 else "")
            )
            
            debug_steps.append({
                "step": f"replacement_level_{level}", 
                "thinking": iter_thinking[:MAX_DEBUG_LEN] + "...",
                "raw_response": raw_resp_iter[:MAX_DEBUG_LEN] + "..."
            })

            if not replace_result or "Q_new" not in replace_result or replace_result['Q_new'] is None:
                return {"error": f"Step {level} rejected/failed", "debug_info": debug_steps, "context": lean_context_for_error}

            current_question = replace_result['Q_new']
            
            for phrase in BANNED_PHRASES:
                if phrase in current_question and phrase not in str(item):
                    return {"error": f"Hallucinated example content: {phrase}", "debug_info": debug_steps, "context": lean_context_for_error}

            final_result["Q_chain"].append({"level": level, "question": current_question})

        # 步骤 5: 成功返回 - [一致性修正完成]
        
        context_to_save = copy.deepcopy(item)
        
        # 遍历 Source Tracking 记录，精准还原模型看到的上下文
        for record in source_tracking:
            key = record['key']
            content = record['content']
            data_type = record['type']
            
            if data_type == 'R':
                # 如果当时用的是 R，就覆写 R (保持 List 结构)
                context_to_save[key] = [content]
                
                # [优化] 清空对应的 Document 文本以节省空间，同时标明引用来源
                doc_key_ref = key.replace("R", "document")
                if doc_key_ref in context_to_save and isinstance(context_to_save[doc_key_ref], dict):
                    context_to_save[doc_key_ref]['text'] = "[Reference R]" 
                    
            elif data_type == 'doc':
                # 如果当时用的是 Doc，就覆写 Doc
                if key in context_to_save and isinstance(context_to_save[key], dict):
                     context_to_save[key]['text'] = content

        final_result["context"] = context_to_save
        final_result["debug_info"] = debug_steps
        return final_result

    except Exception as e:
        return {"error": str(e), "context": lean_context_for_error, "debug_info": debug_steps}

# --- 主流程 ---

class QAGenerator:
    def __init__(self, config):
        self.config = config

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
        
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.failed_output_path), exist_ok=True)

        prompt1_content = load_prompt_template(self.config.prompt1_file)
        prompt2_content = load_prompt_template(self.config.prompt2_file)

        logger.info(f"Starting pool with {self.config.num_workers} workers...")
        
        with Pool(processes=self.config.num_workers, 
                  initializer=worker_initializer, 
                  initargs=(self.config, prompt1_content, prompt2_content)) as pool:
            
            with open(self.config.output_path, "a", encoding='utf-8') as f_success, \
                 open(self.config.failed_output_path, "w", encoding='utf-8') as f_failed:
                
                for result in tqdm(pool.imap_unordered(process_single_item_task, process_args), total=total_items):
                    if "error" in result:
                        f_failed.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        f_success.write(json.dumps(result, ensure_ascii=False) + "\n")
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