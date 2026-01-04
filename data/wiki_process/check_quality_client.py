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

def extract_questions(q_chain: List[Dict[str, Any]]) -> Dict[str, str]:
    """辅助函数：提取 L2, L1, L0 问题"""
    return {
        "L2": next((q["question"] for q in q_chain if q.get("level") == 2), ""),
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
    检查数据是否已存在。
    修改点 1: 只检查 history_file，不检查 current_output。
    修改点 2: 只比对 Question，不比对 Answer。
    """
    
    # 如果没有提供历史文件，或者历史文件不存在，直接认为没跑过
    if not history_file or not os.path.exists(history_file):
        return False

    if not new_data: return False

    # --- 步骤 A: 生成指纹 (随机抽样, 只取 Question) ---
    real_sample_size = min(len(new_data), sample_size)
    sample_records = random.sample(new_data, real_sample_size)
    
    # 使用集合存储待查找的指纹，这里只存 L2 Question 字符串
    pending_fingerprints = set()
    
    for item in sample_records:
        qs = extract_questions(item.get("Q_chain", []))
        q_text = qs.get("L2", "").strip()
        if q_text:
            pending_fingerprints.add(q_text)
            
    if not pending_fingerprints: return False

    # 动态调整阈值
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
                    # 提取已有数据的 Question
                    ex_q = existing.get("question", "").strip()
                    
                    if not ex_q: continue

                    # 检查是否命中 (只比对 Q)
                    if ex_q in pending_fingerprints:
                        match_count += 1
                        # 移除已命中的，避免重复计数
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
# 🔥 核心功能 2: Prompt 构建
# ==========================================

def construct_substitution_check_prompt(complex_q: str, simple_q: str, bridge_entity: str, ref_text: str) -> str:
    return f"""# Role
你是一个极其严格的逻辑校验专家。你的任务是揪出多跳问答中**具有歧义**的实体替换。

# Context
我们通过将“简单问题”中的 [Bridge Entity] 替换为一段描述，生成了“复杂问题”。
我们需要确保：这段描述在结合 [Reference Text] 后，**只能**指向 [Bridge Entity]，而不能指向其他人或物。

# Data
1. **Complex Question**: "{complex_q}"
2. **Simple Question**: "{simple_q}"
3. **Bridge Entity**: "{bridge_entity}"
4. **Reference Text**: 
   "{ref_text}"

# Critical Instructions

1. **Step 1: Extract Phrase**
   找出 Complex Question 中用来指代 Bridge Entity 的描述短语。

2. **Step 2: Verify Factuality**
   验证描述是否符合事实。

3. **Step 3: Check Uniqueness (排他性检查)**
   检查原文中是否有**其他实体**也符合该描述。如果原文列举了多个同类项，且描述无法唯一锁定目标，必须判为 False。

# Output Format (JSON Only)
```json
{{
  "diff_phrase": "...",
  "analysis": "...",
  "valid": true/false
}}
```"""

async def call_llm_json(session, url, model_name, prompt):
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content'].strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                    else:
                        return None
            return None
    except Exception:
        return None

# ==========================================
# 🔥 核心功能 3: 流式写入器
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

            qs = extract_questions(item.get("Q_chain", []))
            
            if item.get("_is_valid_llm", False):
                valid_count += 1
                context = item["context"]
                doc_ref_ids = []
                
                for dk, rk, rt in [("document1", "R1", "R1"), ("document2", "R2", "R2"), ("document3", "R3", "R3")]:
                    title = context.get(dk, {}).get("title", "").strip()
                    raw = context[rk][0] if isinstance(context[rk], list) else str(context[rk])
                    val = f"{title}\n{raw}" if title else raw
                    
                    ref_uuid = str(uuid.uuid4())
                    ref_record = {
                        "id": ref_uuid,
                        "value": val,
                        "meta_data": {
                            "r_type": rt,
                            "document_id": context.get(dk, {}).get("id", ""),
                            "document_title": title,
                            "original_data_source": source_name
                        }
                    }
                    json.dump(ref_record, f_valid_r, ensure_ascii=False)
                    f_valid_r.write('\n')
                    doc_ref_ids.append(ref_uuid)

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
                    "question": qs["L2"],
                    "label": doc_ref_ids,
                    "meta_data": {
                        "connection_d1_d2": fmt_conn("connection_d1_d2"),
                        "connection_d2_d3": fmt_conn("connection_d2_d3"),
                        "question_level_1": qs["L1"],
                        "question_level_0": qs["L0"],
                        "original_data_source": source_name
                    }
                }
                json.dump(q_record, f_valid_q, ensure_ascii=False, separators=(',', ':'))
                f_valid_q.write('\n')
                
            else:
                invalid_count += 1
                simple_record = {
                    "question": qs["L2"],
                    "question_level_1": qs["L1"],
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
# 🔥 核心功能 4: 生产者 Worker
# ==========================================

def get_doc_text(context, doc_key, content_key):
    try:
        title = context.get(doc_key, {}).get("title", "")
        content_raw = context.get(content_key, "")
        if isinstance(content_raw, list):
            content_raw = " ".join(content_raw)
        return f"Title: {title}\nContent: {content_raw}"
    except:
        return ""

async def worker(session, url, model_name, item, semaphore, queue):
    q_chain = item.get("Q_chain", [])
    qs = extract_questions(q_chain)
    context = item.get("context", {})
    
    bridge_2 = context.get("connection_d2_d3", {}).get("bridge", None)
    bridge_1 = context.get("connection_d1_d2", {}).get("bridge", None)
    
    ref_text_step_1 = get_doc_text(context, "document2", "R2")
    ref_text_step_2 = get_doc_text(context, "document1", "R1")

    is_valid = False
    reason = "Data Missing"

    if qs["L2"] and qs["L1"] and qs["L0"] and bridge_1 and bridge_2 and ref_text_step_1 and ref_text_step_2:
        async with semaphore:
            # Step 1
            prompt_step_1 = construct_substitution_check_prompt(
                complex_q=qs["L1"], 
                simple_q=qs["L0"], 
                bridge_entity=bridge_2,
                ref_text=ref_text_step_1
            )
            res_1 = await call_llm_json(session, url, model_name, prompt_step_1)
            
            valid_1 = False
            reason_1 = "LLM Error (Step 1)"
            if res_1:
                val = res_1.get("valid", False)
                if isinstance(val, bool): valid_1 = val
                elif isinstance(val, str): valid_1 = val.lower() == 'true'
                reason_1 = f"[{res_1.get('diff_phrase')}] {res_1.get('analysis')}"

            if not valid_1:
                is_valid = False
                reason = f"Step1 Fail: {reason_1}"
            else:
                # Step 2
                prompt_step_2 = construct_substitution_check_prompt(
                    complex_q=qs["L2"], 
                    simple_q=qs["L1"], 
                    bridge_entity=bridge_1,
                    ref_text=ref_text_step_2
                )
                res_2 = await call_llm_json(session, url, model_name, prompt_step_2)
                
                valid_2 = False
                reason_2 = "LLM Error (Step 2)"
                if res_2:
                    val = res_2.get("valid", False)
                    if isinstance(val, bool): valid_2 = val
                    elif isinstance(val, str): valid_2 = val.lower() == 'true'
                    reason_2 = f"[{res_2.get('diff_phrase')}] {res_2.get('analysis')}"
                
                if valid_2:
                    is_valid = True
                    reason = "Passed Both Steps"
                else:
                    is_valid = False
                    reason = f"Step2 Fail: {reason_2}"

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

    # === [UPDATE] 去重检查 ===
    # 只检查 history_file，只比对 Q，不检查 current_output
    if pre_check_duplication(data, 
                             history_file=args.history_file, 
                             sample_size=100, 
                             match_threshold=5):
        print(f"⏭️  [Skip] 数据已在历史文件 [{args.history_file}] 中检测到，跳过处理。")
        return

    # 基础过滤
    print("步骤 1/2: 基础过滤 (Answer Leakage)...")
    candidates = []
    invalid_step1 = []
    
    for item in data:
        ans = item.get("A", "").strip()
        qs = extract_questions(item.get("Q_chain", []))
        leaked = False
        for q_txt in qs.values():
            if ans in q_txt:
                leaked = True; break
        
        if not leaked:
            candidates.append(item)
        else:
            invalid_step1.append({
                "question": qs["L2"], "answer": ans, "invalid_reason": "Answer Leakage"
            })
            
    if invalid_step1:
        save_lines_append(args.output_invalid, invalid_step1)
    
    print(f"待 vLLM 校验数据: {len(candidates)} 条")
    if not candidates: return

    # 启动处理
    print(f"步骤 2/2: vLLM 分步校验...")
    queue = asyncio.Queue()
    writer_task = asyncio.create_task(result_writer_service(queue, {
        "questions": args.output_questions,
        "references": args.output_references,
        "invalid": args.output_invalid
    }, source_name))

    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, args.api_url, args.model_name, item, semaphore, queue)) for item in candidates]
        for f in tqdm.as_completed(tasks): await f

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
    # === [UPDATE] 新增参数 ===
    parser.add_argument("--history-file", default=None, help="仅用于去重检查的全局历史文件 (.jsonl)")
    args = parser.parse_args()

    try:
        asyncio.run(main_pipeline(args))
    except KeyboardInterrupt:
        print("\n⛔ 用户中断。")

if __name__ == "__main__":
    main()