import json
import argparse
import os
import random
from typing import List, Dict, Any, Optional

def load_original_data(input_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式原始数据，过滤无效行（非大括号起始、解析失败、字段缺失）"""
    original_data = []
    valid_lines = 0
    invalid_lines = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                cleaned_line = line.strip()
                
                # 过滤空行
                if not cleaned_line:
                    invalid_lines += 1
                    continue
                
                # 过滤非大括号起始的行
                if not cleaned_line.startswith('{'):
                    print(f"警告：第{line_num}行非大括号起始，已忽略 -> 内容：{cleaned_line[:50]}...")
                    invalid_lines += 1
                    continue
                
                # 解析JSON
                try:
                    data_item = json.loads(cleaned_line)
                except json.JSONDecodeError as e:
                    print(f"警告：第{line_num}行JSON格式错误，已忽略 -> 错误：{str(e)[:50]}")
                    invalid_lines += 1
                    continue
                
                # 校验必要字段
                required_fields = ["A", "Q_chain", "context"]
                if not all(field in data_item for field in required_fields):
                    print(f"警告：第{line_num}行缺失必要字段（A/Q_chain/context），已忽略")
                    invalid_lines += 1
                    continue
                
                # 校验context中的必要字段
                context = data_item["context"]
                required_context_fields = ["R1", "R2", "R3", "connection_d1_d2", "connection_d2_d3"]
                if not all(field in context for field in required_context_fields):
                    print(f"警告：第{line_num}行context缺失必要字段（R1/R2/R3/connection_d1_d2/connection_d2_d3），已忽略")
                    invalid_lines += 1
                    continue
                
                original_data.append(data_item)
                valid_lines += 1

        print(f"\n数据加载完成：")
        print(f"- 总行数：{line_num}")
        print(f"- 有效行数（基础过滤后）：{valid_lines}")
        print(f"- 无效行数（格式/字段问题）：{invalid_lines}")
        return original_data

    except FileNotFoundError:
        raise FileNotFoundError(f"未找到输入文件：{input_path}")
    except Exception as e:
        raise RuntimeError(f"加载文件时发生未知错误：{str(e)}")

def is_answer_in_question(answer: str, question: str) -> bool:
    """判断answer是否完整出现在question中（去前后空格后匹配）"""
    if not answer or not question:
        return False  # 任一为空，不视为匹配
    # 去前后空格后判断子串包含（完整出现）
    clean_answer = answer.strip()
    clean_question = question.strip()
    return clean_answer in clean_question

def filter_invalid_data(original_data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    过滤无效数据：
    1. Answer完整出现在任何一轮Question中
    2. 任何一轮Question包含模糊关键词（周围、附近、之一等）
    返回：(有效数据列表, 无效数据列表)
    """
    filtered_data = []
    invalid_data = []
    
    # 定义需要过滤的模糊关键词（及其相似表述）
    VAGUE_KEYWORDS = ["周围", "附近", "之一", "旁边", "邻近"]

    for idx, item in enumerate(original_data, 1):
        answer = item.get("A", "").strip()
        q_chain = item.get("Q_chain", [])
        
        # 基础检查：必须存在 level=2 的 question 才能作为有效数据的基础
        has_level2 = False
        for q in q_chain:
            if q.get("level") == 2:
                has_level2 = True
                break
        
        if not has_level2:
            print(f"警告：第{idx}条数据未找到level2的question，已跳过")
            continue

        # --- 新增过滤逻辑 ---
        is_invalid = False
        invalid_reason = ""

        for q_item in q_chain:
            question_text = q_item.get("question", "").strip()
            level = q_item.get("level")
            
            # 规则 1: Answer 出现在该轮 Question 中
            if is_answer_in_question(answer, question_text):
                is_invalid = True
                invalid_reason = f"Answer完整出现在level={level}的Question中"
                break
            
            # 规则 2: 该轮 Question 包含模糊关键词
            for kw in VAGUE_KEYWORDS:
                if kw in question_text:
                    is_invalid = True
                    invalid_reason = f"Question(level={level})包含模糊关键词'{kw}'"
                    break
            
            if is_invalid:
                break # 只要发现一个问题满足条件，整组数据即无效
        
        if is_invalid:
            item["invalid_reason"] = invalid_reason
            item["invalid_data_source"] = input_path.split('/')[-2]
            invalid_data.append(item)
            print(f"过滤无效数据：第{idx}条 -> {invalid_reason}")
        else:
            filtered_data.append(item)
    
    print(f"\n数据过滤完成：")
    print(f"- 过滤后有效数据数：{len(filtered_data)}")
    print(f"- 无效数据数：{len(invalid_data)}")
    return filtered_data, invalid_data

def save_invalid_data(invalid_data: List[Dict[str, Any]], invalid_output_path: str):
    """保存无效数据到独立JSONL文件（追加模式）"""
    if not invalid_data:
        return
    
    # 追加写入JSONL格式
    with open(invalid_output_path, 'a+', encoding='utf-8') as f:
        for record in invalid_data:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    # 统计无效文件总记录数
    total_invalid = sum(1 for line in open(invalid_output_path, 'r', encoding='utf-8') if line.strip())
    print(f"\n无效数据已追加至：{invalid_output_path}")
    print(f"无效文件当前总记录数：{total_invalid}")

def get_max_existing_r_count(reference_path: str) -> int:
    """读取JSONL格式参考文件的有效行数（即已有的R总数），作为起始编号依据"""
    max_count = 0
    if os.path.exists(reference_path):
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue
                    try:
                        json.loads(cleaned_line)
                        max_count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"参考文件已存在 {max_count} 个R记录（最大编号：{max_count}）")
        except Exception as e:
            print(f"读取参考文件失败：{str(e)}，将从1开始分配R编号")
            max_count = 0
    else:
        print(f"参考文件不存在，将从1开始分配R编号")
    return max_count

def format_connection(connection: Dict[str, Any]) -> Dict[str, Any]:
    """格式化connection字段：移除edit_distance，similarity保留两位小数"""
    formatted = {}
    if "bridge" in connection:
        formatted["bridge"] = connection["bridge"]
    if "similarity" in connection:
        try:
            similarity = float(connection["similarity"])
            formatted["similarity"] = round(similarity, 2)
        except (ValueError, TypeError):
            formatted["similarity"] = connection["similarity"]
    return formatted

def process_r_reference(original_data: List[Dict[str, Any]], start_r_id: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    处理R数据和编号分配（从start_r_id开始）
    包含功能：
    1. R内容前拼接Title
    2. 提取Level 0/1问题到Metadata
    """
    current_new_r_id = start_r_id
    reference_records = []
    data_r_mapping = []

    for idx, item in enumerate(original_data):
        context = item["context"]
        r_ids = []

        # 辅助函数：生成带标题的内容
        def format_r_content(doc_key, r_key):
            doc_info = context.get(doc_key, {})
            title = doc_info.get("title", "").strip()
            raw_content = context[r_key][0] if context[r_key] and isinstance(context[r_key], list) else str(context[r_key])
            
            if title:
                return f"{title}\n{raw_content}", title, doc_info.get("id", "")
            else:
                return raw_content, title, doc_info.get("id", "")

        # --- 处理R1, R2, R3 ---
        for doc_key, r_key, r_type in [
            ("document1", "R1", "R1"),
            ("document2", "R2", "R2"),
            ("document3", "R3", "R3")
        ]:
            val, title, doc_id = format_r_content(doc_key, r_key)
            record = {
                "value": val,
                "meta_data": {
                    "r_type": r_type,
                    "document_id": doc_id,
                    "document_title": title,
                    "original_data_source": input_path.split('/')[-2]
                }
            }
            reference_records.append(record)
            r_ids.append(current_new_r_id)
            current_new_r_id += 1

        data_r_mapping.append(r_ids)

    # 生成问题文件数据
    questions_data = []
    for idx, item in enumerate(original_data):
        # 提取各级问题
        question_lvl_2 = None
        question_lvl_1 = None
        question_lvl_0 = None
        
        for q in item["Q_chain"]:
            level = q.get("level")
            if level == 2:
                question_lvl_2 = q["question"]
            elif level == 1:
                question_lvl_1 = q["question"]
            elif level == 0:
                question_lvl_0 = q["question"]
        
        if not question_lvl_2:
            continue
        
        context = item["context"]
        connection_d1_d2 = format_connection(context.get("connection_d1_d2", {}))
        connection_d2_d3 = format_connection(context.get("connection_d2_d3", {}))
        
        questions_data.append({
            "answer": item["A"],
            "question": question_lvl_2,
            "label": data_r_mapping[idx],
            "meta_data": {
                "connection_d1_d2": connection_d1_d2,
                "connection_d2_d3": connection_d2_d3,
                "question_level_1": question_lvl_1,
                "question_level_0": question_lvl_0,
                "original_data_source": input_path.split('/')[-2]
            }
        })

    print(f"\nR数据处理完成：")
    print(f"- 新增R记录数：{len(reference_records)}")
    print(f"- 生成有效问题数据数：{len(questions_data)}")
    print(f"- 本次分配R编号范围：{start_r_id} ~ {current_new_r_id - 1}")
    return questions_data, reference_records

def sample_random_records(records: List[Dict[str, Any]], sample_size: int = 3) -> List[Dict[str, Any]]:
    """随机抽样用于重复检测"""
    if len(records) <= sample_size:
        return records.copy()
    return random.sample(records, sample_size)

def check_duplicate_qa(qa_path: str, sample_records: List[Dict[str, Any]]) -> bool:
    """检查抽样数据是否已存在"""
    if not os.path.exists(qa_path):
        return False
    
    sample_keys = []
    for record in sample_records:
        key = (record.get("question", "").strip(), record.get("answer", "").strip())
        sample_keys.append(key)
    
    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                cleaned_line = line.strip()
                if not cleaned_line:
                    continue
                try:
                    existing = json.loads(cleaned_line)
                    existing_key = (existing.get("question", "").strip(), existing.get("answer", "").strip())
                    if existing_key in sample_keys:
                        print(f"❌ 发现重复数据！现有文件第{line_num}行与抽样数据重复")
                        print(f"  - 重复question：{existing.get('question', '')[:50]}...")
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"⚠️  读取现有QA文件时发生错误：{str(e)}，跳过重复检测")
        return False
    
    return False

def save_output_files(questions_data: List[Dict[str, Any]], new_reference_records: List[Dict[str, Any]],
                     q_output_path: str, r_output_path: str):
    """保存结果文件"""
    with open(q_output_path, 'a+', encoding='utf-8') as f:
        for record in questions_data:
            json.dump(record, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    print(f"\nQA文件已追加至：{q_output_path}")
    qa_total = sum(1 for line in open(q_output_path, 'r', encoding='utf-8') if line.strip())
    print(f"QA文件当前总记录数：{qa_total}")

    with open(r_output_path, 'a+', encoding='utf-8') as f:
        for record in new_reference_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print(f"参考文件已追加至：{r_output_path}")
    ref_total = get_max_existing_r_count(r_output_path)
    print(f"参考文件当前总记录数：{ref_total}")

def main():
    parser = argparse.ArgumentParser(description="JSONL数据处理：深度过滤+重复检测+R处理")
    parser.add_argument("--input", required=True, help="原始数据路径")
    parser.add_argument("--output-questions", required=True, help="全局QA输出路径")
    parser.add_argument("--output-references", required=True, help="全局R输出路径")
    parser.add_argument("--output-invalid", required=True, help="无效数据输出路径")
    args = parser.parse_args()

    global input_path
    input_path = args.input

    try:
        original_data = load_original_data(args.input)
        if not original_data:
            return
        
        print("\n==================================================")
        print("🔍 开始深度过滤无效数据（全链路Answer检测 + 模糊词过滤）...")
        print("==================================================")
        filtered_data, invalid_data = filter_invalid_data(original_data)
        
        save_invalid_data(invalid_data, args.output_invalid)
        
        if not filtered_data:
            print("警告：过滤后无有效数据，程序结束")
            return
        
        existing_count = get_max_existing_r_count(args.output_references)
        start_r_id = existing_count + 1
        
        questions_data, new_reference_records = process_r_reference(filtered_data, start_r_id)
        if not questions_data:
            return
        
        print("\n==================================================")
        print("🔍 开始重复检测...")
        print("==================================================")
        sample_records = sample_random_records(questions_data)
        if check_duplicate_qa(args.output_questions, sample_records):
            print("❌ 错误：数据重复，取消保存")
            return
        
        print("\n==================================================")
        print("💾 开始保存有效文件...")
        print("==================================================")
        save_output_files(questions_data, new_reference_records, args.output_questions, args.output_references)
        
        print("\n✅ 处理完全完成！")
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        raise

if __name__ == "__main__":
    main()