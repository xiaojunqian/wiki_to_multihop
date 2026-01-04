# -*- coding: utf-8 -*-
"""
多跳QA数据集生成器 (三步Prompt版 + 防复读 + 严格一致性校验)
功能：
1. 推理时使用完整文档片段或拼接片段 (final_r_list)。
2. 使用三步 Prompt 策略 (Check Same -> Generate -> Check Unique) 提高生成质量。
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
WORKER_PROMPT_SAME = ""
WORKER_PROMPT_GEN = ""
WORKER_PROMPT_UNIQUE = ""
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

def worker_initializer(args, prompt1_text, prompt_same_text, prompt_gen_text, prompt_unique_text):
    global WORKER_CLIENTS, WORKER_CONFIG, WORKER_PROMPT1, WORKER_PROMPT_SAME, WORKER_PROMPT_GEN, WORKER_PROMPT_UNIQUE, WORKER_GEN_CFG
    
    WORKER_CONFIG = args
    WORKER_PROMPT1 = prompt1_text
    WORKER_PROMPT_SAME = prompt_same_text
    WORKER_PROMPT_GEN = prompt_gen_text
    WORKER_PROMPT_UNIQUE = prompt_unique_text
    
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
    # 处理 Split Marker (System / User 分离)
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
    global WORKER_CLIENTS, WORKER_PROMPT1, WORKER_PROMPT_SAME, WORKER_PROMPT_GEN, WORKER_PROMPT_UNIQUE
    
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
    source_tracking = [] 

    i = 1
    while f'document{i}' in item:
        r_list_raw.append(item[f'document{i}']['text'])
        i += 1
    
    for idx in range(len(r_list_raw)):
        doc_id = idx + 1
        r_key = f'R{doc_id}'
        doc_key = f'document{doc_id}'
        
        if r_key in item and item[r_key] and isinstance(item[r_key], list):
            try:
                if len(item[r_key]) == 1:
                    content = item[r_key][0]
                else:
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
            content = r_list_raw[idx]
            final_r_list.append(content)
            source_tracking.append({"type": "doc", "key": doc_key, "content": content})

    b_list = []
    i = 1
    while f'connection_d{i}_d{i+1}' in item:
        b_list.append(item[f'connection_d{i}_d{i+1}']['bridge'])
        i += 1

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
        # === 阶段 1: 初始问题生成 (LLM Init) ===
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

        # === 阶段 2: 三步迭代生成 (Loop) ===
        # i=1: 替换 b_list[1] (Bridge2), 使用 final_r_list[1] (Doc2), Bridge是 b_list[0]
        
        for i in reversed(range(len(b_list))):
            level = len(b_list) - i
            target_entity = b_list[i]
            if i > 0:
                occupy_entity = b_list[i-1]
            else:
                occupy_entity = ""
            context_document = final_r_list[i]
            last_document = final_r_list[i+1] # 作为一致性检查的 Ref
            
            # --- Sub-step 2.1: Consistency Check ---
            check_same_res, same_think, same_raw = _call_llm_step(
                client, WORKER_PROMPT_SAME,
                target_entity=target_entity,
                current_question=current_question,
                last_document=last_document,
                context_document=context_document
            )
            debug_steps.append({"step": f"L{level}_check_same", "response": check_same_res})
            
            # 兼容 key 名: is_same_entity 或 is_different_entity (取反)
            is_same = check_same_res.get('is_same_entity', False)
            if not is_same:
                 # 容错：如果 JSON 解析失败或者明确判定不同，则拒绝
                 return {"error": f"Step {level} Check Same Failed", "debug_info": debug_steps, "context": lean_context_for_error}

            # --- Sub-step 2.2: Generate Description ---
            gen_res, gen_think, gen_raw = _call_llm_step(
                client, WORKER_PROMPT_GEN,
                target_entity=target_entity,
                occupy_entity=occupy_entity,
                context_document=context_document,
                current_question=current_question
            )
            debug_steps.append({"step": f"L{level}_generate", "response": gen_res})
            
            generated_description = gen_res.get('generated_description', "")
            if not generated_description:
                return {"error": f"Step {level} Gen Failed (Empty)", "debug_info": debug_steps}

            # --- Sub-step 2.3: Uniqueness Check ---
            unique_res, unique_think, unique_raw = _call_llm_step(
                client, WORKER_PROMPT_UNIQUE,
                target_entity=target_entity,
                generated_description=generated_description,
                context_document=context_document,
                current_question=current_question
            )
            debug_steps.append({"step": f"L{level}_check_unique", "response": unique_res})
            
            if not unique_res or not unique_res.get('is_valid', False):
                return {"error": f"Step {level} Check Unique Failed", "debug_info": debug_steps}

            if unique_res.get('rephrased_question'):
                # 直接采纳 AI 润色后的结果
                current_question = unique_res['rephrased_question']
            else:
                # 兜底逻辑：如果 AI 偷懒没返回句子，还是执行原来的硬替换
                logger.warning(f"Model valid but no rephrased_question at level {level}, using fallback replace.")
                if target_entity not in current_question:
                    return {"error": f"Target '{target_entity}' missing in fallback replace", "debug_info": debug_steps}
                current_question = current_question.replace(target_entity, generated_description)

            # 防复读检查 (保持不变)
            for phrase in BANNED_PHRASES:
                if phrase in current_question and phrase not in str(item):
                    return {"error": f"Hallucinated example content: {phrase}", "debug_info": debug_steps, "context": lean_context_for_error}

            final_result["Q_chain"].append({"level": level, "question": current_question})

        # 步骤 3: 成功返回 - [一致性修正]
        context_to_save = copy.deepcopy(item)
        for record in source_tracking:
            key = record['key']
            content = record['content']
            data_type = record['type']
            if data_type == 'R':
                context_to_save[key] = [content]
                doc_key_ref = key.replace("R", "document")
                if doc_key_ref in context_to_save and isinstance(context_to_save[doc_key_ref], dict):
                    context_to_save[doc_key_ref]['text'] = "[Reference R]" 
            elif data_type == 'doc':
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

        # 加载所有 Prompt 文件
        prompt1_content = load_prompt_template(self.config.prompt1_file)
        prompt_same_content = load_prompt_template(self.config.prompt_check_same_file)
        prompt_gen_content = load_prompt_template(self.config.prompt_generate_file)
        prompt_unique_content = load_prompt_template(self.config.prompt_check_unique_file)

        logger.info(f"Starting pool with {self.config.num_workers} workers...")
        
        with Pool(processes=self.config.num_workers, 
                  initializer=worker_initializer, 
                  initargs=(self.config, prompt1_content, prompt_same_content, prompt_gen_content, prompt_unique_content)) as pool:
            
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
    
    # Updated Prompt Arguments
    parser.add_argument("--prompt1_file", type=str, required=True)
    parser.add_argument("--prompt_check_same_file", type=str, required=True, help="Path to Check Consistency Prompt")
    parser.add_argument("--prompt_generate_file", type=str, required=True, help="Path to Generate Description Prompt")
    parser.add_argument("--prompt_check_unique_file", type=str, required=True, help="Path to Check Uniqueness Prompt")
    
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