import json
import argparse
import os
import random
import asyncio
import aiohttp
import re
import uuid
import sys
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm

# ==========================================
# 基础工具模块
# ==========================================

def load_original_data(input_path: str) -> List[Dict[str, Any]]:
    """加载原始数据 (JSONL)"""
    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('{'):
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return []

def save_lines_append(file_path: str, data_list: List[Dict[str, Any]]):
    """通用追加写入工具"""
    if not data_list:
        return
    with open(file_path, 'a+', encoding='utf-8') as f:
        for item in data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def extract_questions_1hop(q_chain: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    [L1 核心逻辑] 
    L1 = 1-hop问题 (核心验证对象)
    L0 = 基础问题
    """
    return {
        "L1": next((q["question"] for q in q_chain if q.get("level") == 1), ""),
        "L0": next((q["question"] for q in q_chain if q.get("level") == 0), "")
    }

# ==========================================
# 🔥 核心功能 1: 局部重复预检 (Q-Only, History-Only)
# ==========================================

def pre_check_duplication(new_data: List[Dict[str, Any]], 
                          history_file: str = None, 
                          sample_size: int = 100, 
                          match_threshold: int = 10) -> bool:
    """
    与 2-hop 代码保持一致：
    1. 只检查 history_file。
    2. 只比对 L1 Question 文本。
    """
    
    if not history_file or not os.path.exists(history_file):
        return False

    if not new_data: return False

    # --- 步骤 A: 生成指纹 (随机抽样, 只取 L1 Question) ---
    real_sample_size = min(len(new_data), sample_size)
    sample_records = random.sample(new_data, real_sample_size)
    
    pending_fingerprints = set()
    
    for item in sample_records:
        qs = extract_questions_1hop(item.get("Q_chain", []))
        q_text = qs.get("L1", "").strip() # 这里取 L1
        if q_text:
            pending_fingerprints.add(q_text)
            
    if not pending_fingerprints: return False

    current_threshold = min(match_threshold, len(pending_fingerprints))

    print(f"🔍 [Pre-Check] 正在扫描历史文件 [{os.path.basename(history_file)}] ...")
    print(f"   - 采样指纹(Q only): {len(pending_fingerprints)} 条")
    print(f"   - 判定阈值: 命中 >= {current_threshold} 条即视为重复")

    match_count = 0

    # --- 步骤 B: 扫描历史文件 ---
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    existing = json.loads(line)
                    ex_q = existing.get("question", "").strip()
                    if not ex_q: continue

                    if ex_q in pending_fingerprints:
                        match_count += 1
                        pending_fingerprints.remove(ex_q)
                        
                        if match_count >= current_threshold:
                            print(f"⏭️  [Duplicate] 命中次数达到 {match_count}，判定任务已运行过。")
                            return True 
                except:
                    continue
    except Exception as e:
        print(f"⚠️  [Pre-Check] 读取文件 {history_file} 出错: {e}，跳过检查")
        return False

    print(f"   [Pre-Check] 最终命中: {match_count}/{current_threshold}。未达到阈值，视为新任务。")
    return False

# ==========================================
# 🔥 核心功能 2: Prompt 构建 (L1 验证)
# ==========================================

def construct_1hop_validation_prompt(qs: Dict[str, str], bridge: str) -> str:
    """
    [保持不变] 验证 L1 是否唯一指代 Bridge
    """
    return f"""# Role
你是一个严格的逻辑检查员。我们需要验证一个单跳问答推理链条中的指代是否**精准且唯一**。

# Context
我们有两层问题 (L0 -> L1) 和一个连接它们的“桥梁实体”。
- **L0 (基础问题)**: {qs['L0']}
- **L1 (1-hop问题)**: {qs['L1']}

# Bridge Entity
- **Bridge**: {bridge}

# Task
请**一步步思考**，检查下面的逻辑连接是否有效：

**Check (L1 vs L0)**: L1 中多出的描述（相较于 L0）是否**唯一**地指代了 `Bridge` 实体？

# Criteria 
- **合格 (true)**: 
  - 描述具有唯一指向性（如“...的首府”、“...的导演”、“...的出生年份”）。
  - 允许措辞的细微变化（如“执导的人”vs“导演”）。
  - 如果你的判断是** L1中没有提到xxx，因此无法唯一指代Bridge **，如果L1指向没有简单的歧义，仍然判断为合格。
  - 外部信息导致，描述是一个指向唯一确定的名词，默认能正确指向对应的Bridge。
- **不合格 (false)**: 
  - ** 模糊指代**: 只有“...之一”、“...的一个城市”、“...的成员”、“...的一部分”。
  - ** 逻辑错误**: 描述完全错误。
  - ** 原地指代**: 用了与之前字词高度相似的描述（例如 L0 问“X在哪里”，L1 问“X的位置在哪里”，没有实质替换）。
  - ** 注意: 只有在明显指代有歧义，或者明显有错误时才输出false。
  
# Output Format (JSON Only)
不要输出任何闲聊，严格按照以下 JSON 格式输出，必须包含 valid 字段：
```json
{{
  "reason": "简短分析L1对Bridge的指代情况...",
  "valid": true/false
}}
```"""

async def call_llm_json(session, url, model_name, prompt):
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        async with session.post(url, json=payload, timeout=60) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content'].strip()
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '')
                try:
                    return json.loads(content)
                except:
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    return json.loads(match.group(0)) if match else None
            return None
    except Exception:
        return None

# ==========================================
# 🔥 核心功能 3: 写入器 (Writer) - 格式已统一
# ==========================================

async def result_writer_service(queue: asyncio.Queue, file_paths: Dict[str, str], source_name: str):
    f_valid_q = open(file_paths["questions"], 'a+', encoding='utf-8')
    f_valid_r = open(file_paths["references"], 'a+', encoding='utf-8')
    f_invalid = open(file_paths["invalid"], 'a+', encoding='utf-8')

    valid_count = 0
    invalid_count = 0

    try:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            qs = extract_questions_1hop(item.get("Q_chain", []))
            
            if item.get("_is_valid_llm", False):
                valid_count += 1
                context = item.get("context", {})
                doc_ref_ids = []
                
                # [Format Update] 统一遍历 document1, document2，并记录 document_id
                # 1-hop 通常涉及 R1, R2
                for dk, rk, rt in [("document1", "R1", "R1"), ("document2", "R2", "R2")]:
                    raw_content = context.get(rk)
                    if not raw_content: continue
                    
                    # 兼容 list 或 str
                    val = raw_content[0] if isinstance(raw_content, list) and len(raw_content) > 0 else str(raw_content)
                    
                    doc_meta = context.get(dk, {})
                    title = doc_meta.get("title", "").strip()
                    doc_id = doc_meta.get("id", "") # 获取 document_id
                    
                    full_val = f"{title}\n{val}" if title else val

                    ref_uuid = str(uuid.uuid4())
                    ref_record = {
                        "id": ref_uuid,
                        "value": full_val,
                        "meta_data": {
                            "r_type": rt,
                            "document_id": doc_id,    # [统一]
                            "document_title": title,  # [统一]
                            "original_data_source": source_name
                        }
                    }
                    json.dump(ref_record, f_valid_r, ensure_ascii=False)
                    f_valid_r.write('\n')
                    doc_ref_ids.append(ref_uuid)

                # [Format Update] 统一 connection 结构
                def fmt_conn(key):
                    c = context.get(key, {})
                    res = {}
                    if "bridge" in c: res["bridge"] = c["bridge"]
                    if "similarity" in c:
                        try: res["similarity"] = round(float(c["similarity"]), 2)
                        except: res["similarity"] = c["similarity"]
                    return res

                q_record = {
                    "answer": item["A"],
                    "question": qs["L1"], # 1-hop 的顶层问题是 L1
                    "label": doc_ref_ids,
                    "meta_data": {
                        # 1-hop 只有 d1-d2 的连接
                        "connection_d1_d2": fmt_conn("connection_d1_d2"), 
                        "question_level_0": qs["L0"], # [统一] 重命名为 question_level_0
                        "original_data_source": source_name
                    }
                }
                json.dump(q_record, f_valid_q, ensure_ascii=False, separators=(',', ':'))
                f_valid_q.write('\n')
                
            else:
                invalid_count += 1
                simple_record = {
                    "question": qs["L1"], 
                    "question_level_0": qs["L0"],
                    "answer": item.get("A", ""),
                    "invalid_reason": item.get("_llm_reason", "LLM Check Failed")
                }
                json.dump(simple_record, f_invalid, ensure_ascii=False)
                f_invalid.write('\n')

            queue.task_done()
            if (valid_count + invalid_count) % 10 == 0:
                f_valid_q.flush()
                f_valid_r.flush()
                f_invalid.flush()

    finally:
        f_valid_q.close()
        f_valid_r.close()
        f_invalid.close()
        print(f"\n📊 写入完成: 有效 {valid_count} 条, 无效 {invalid_count} 条")

# ==========================================
# 🔥 核心功能 4: Worker
# ==========================================

async def worker(session, url, model_name, item, semaphore, queue):
    q_chain = item.get("Q_chain", [])
    qs = extract_questions_1hop(q_chain)
    context = item.get("context", {})
    target_bridge = context.get("connection_d1_d2", {}).get("bridge", None)

    is_valid = False
    reason = "Data Missing"

    # [验证逻辑] L1, L0, Bridge 必须都存在
    if qs["L1"] and qs["L0"] and target_bridge:
        prompt = construct_1hop_validation_prompt(qs, target_bridge)
        async with semaphore:
            result_json = await call_llm_json(session, url, model_name, prompt)
            if result_json:
                def check_bool(val):
                    if isinstance(val, bool): return val
                    if isinstance(val, str): return val.lower() == 'true'
                    return False
                is_valid = check_bool(result_json.get("valid", False))
                reason = result_json.get("reason", "Passed" if is_valid else "Unknown")
            else:
                reason = "LLM JSON Error"
    
    item["_is_valid_llm"] = is_valid
    item["_llm_reason"] = reason
    await queue.put(item)

# ==========================================
# 主流程入口
# ==========================================

async def main_pipeline(args):
    try:
        source_name = args.input.split('/')[-2]
    except:
        source_name = "unknown"
    
    print(f"📄 加载数据: {args.input}")
    data = load_original_data(args.input)
    if not data: return

    # [统一] 使用升级版的 pre_check
    if pre_check_duplication(data, 
                             history_file=args.history_file, 
                             sample_size=100, 
                             match_threshold=5):
        print(f"⏭️  [Skip] 数据已在历史文件 [{args.history_file}] 中检测到，跳过处理。")
        return

    # 基础过滤：必须有 L1
    print("步骤 1/2: 基础过滤 (L1 Check)...")
    candidates = []
    for item in data:
        qs = extract_questions_1hop(item.get("Q_chain", []))
        if qs["L1"] and item.get("A"):
            candidates.append(item)
            
    print(f"待校验数据: {len(candidates)} 条")
    if not candidates: return

    print(f"步骤 2/2: LLM 校验...")
    queue = asyncio.Queue()
    file_paths = {
        "questions": args.output_questions,
        "references": args.output_references,
        "invalid": args.output_invalid
    }

    writer_task = asyncio.create_task(
        result_writer_service(queue, file_paths, source_name)
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, args.api_url, args.model_name, i, semaphore, queue)) for i in candidates]
        for f in tqdm.as_completed(tasks):
            await f

    await queue.put(None)
    await writer_task
    print("🎉 任务完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-questions", required=True)
    parser.add_argument("--output-references", required=True)
    parser.add_argument("--output-invalid", required=True)
    parser.add_argument("--api-url", default="http://localhost:8111/v1/chat/completions")
    parser.add_argument("--model-name", default="qwen")
    parser.add_argument("--concurrency", type=int, default=64)
    # [统一] 参数命名保持一致
    parser.add_argument("--history-file", default=None, help="仅用于去重检查的全局历史文件 (.jsonl)")
    args = parser.parse_args()

    try:
        asyncio.run(main_pipeline(args))
    except KeyboardInterrupt:
        print("\n⛔ 用户中断。")

if __name__ == "__main__":
    main()