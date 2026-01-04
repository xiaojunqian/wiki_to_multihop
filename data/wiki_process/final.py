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
    过滤无效数据：answer完整出现在question中的数据
    返回：(有效数据列表, 无效数据列表)
    """
    filtered_data = []
    invalid_data = []

    for idx, item in enumerate(original_data, 1):
        # 先找到level2的question（与后续生成QA的question一致）
        question = None
        for q in item["Q_chain"]:
            if q.get("level") == 2:
                question = q["question"]
                break
        
        if not question:
            # 无level2 question，按之前逻辑跳过，不加入无效数据
            print(f"警告：第{idx}条数据未找到level2的question，已跳过")
            continue
        
        answer = item.get("A", "").strip()
        # 判断是否满足无效条件
        if is_answer_in_question(answer, question):
            # 记录无效原因，便于追溯
            item["invalid_reason"] = f"answer='{answer[:30]}...' 完整出现在 question='{question[:50]}...' 中"
            item["invalid_data_source"] = input_path.split('/')[-2]  # 记录数据来源
            invalid_data.append(item)
            print(f"过滤无效数据：第{idx}条 -> {item['invalid_reason']}")
        else:
            filtered_data.append(item)
    
    print(f"\n数据过滤完成：")
    print(f"- 过滤后有效数据数：{len(filtered_data)}")
    print(f"- 无效数据数（answer在question中）：{len(invalid_data)}")
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
                    # 验证行是否为有效JSON（避免统计无效行）
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
    # 保留bridge字段
    if "bridge" in connection:
        formatted["bridge"] = connection["bridge"]
    # 处理similarity（保留两位小数）
    if "similarity" in connection:
        try:
            similarity = float(connection["similarity"])
            formatted["similarity"] = round(similarity, 2)
        except (ValueError, TypeError):
            formatted["similarity"] = connection["similarity"]  # 保留原始值，避免报错
    return formatted

def process_r_reference(original_data: List[Dict[str, Any]], start_r_id: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    处理R数据和编号分配（从start_r_id开始）
    修改功能：
    1. 在R的value前拼接document_title。
    2. 提取level 0和level 1的问题到meta_data。
    返回：(问题文件数据, 参考文件数据)
    """
    current_new_r_id = start_r_id
    reference_records = []  # 存储JSONL格式的R记录（value + meta_data）
    data_r_mapping = []    # 存储每条数据对应的R编号列表

    for idx, item in enumerate(original_data):
        context = item["context"]
        r_ids = []  # 按R1→R2→R3顺序存储全局唯一编号

        # 辅助函数：生成带标题的内容
        def format_r_content(doc_key, r_key):
            doc_info = context.get(doc_key, {})
            title = doc_info.get("title", "").strip()
            
            raw_content = context[r_key][0] if context[r_key] and isinstance(context[r_key], list) else str(context[r_key])
            
            # 如果有标题，拼接 "标题\n正文"；否则只返回正文
            if title:
                return f"{title}\n{raw_content}", title, doc_info.get("id", "")
            else:
                return raw_content, title, doc_info.get("id", "")

        # --- 处理R1 (document1) ---
        r1_val, r1_title, r1_id = format_r_content("document1", "R1")
        r1_record = {
            "value": r1_val,
            "meta_data": {
                "r_type": "R1",
                "document_id": r1_id,
                "document_title": r1_title,
                "original_data_source": input_path.split('/')[-2]
            }
        }
        reference_records.append(r1_record)
        r_ids.append(current_new_r_id)
        current_new_r_id += 1

        # --- 处理R2 (document2) ---
        r2_val, r2_title, r2_id = format_r_content("document2", "R2")
        r2_record = {
            "value": r2_val,
            "meta_data": {
                "r_type": "R2",
                "document_id": r2_id,
                "document_title": r2_title,
                "original_data_source": input_path.split('/')[-2]
            }
        }
        reference_records.append(r2_record)
        r_ids.append(current_new_r_id)
        current_new_r_id += 1

        # --- 处理R3 (document3) ---
        r3_val, r3_title, r3_id = format_r_content("document3", "R3")
        r3_record = {
            "value": r3_val,
            "meta_data": {
                "r_type": "R3",
                "document_id": r3_id,
                "document_title": r3_title,
                "original_data_source": input_path.split('/')[-2]
            }
        }
        reference_records.append(r3_record)
        r_ids.append(current_new_r_id)
        current_new_r_id += 1

        data_r_mapping.append(r_ids)

    # 生成问题文件数据（answer + question + label + meta_data）
    questions_data = []
    for idx, item in enumerate(original_data):
        # 遍历 Q_chain 获取各层级问题
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
        
        # 如果没有level 2问题，视为无效
        if not question_lvl_2:
            print(f"警告：第{idx+1}条有效数据未找到level=2的question，已跳过")
            continue
        
        # 提取并格式化connection字段
        context = item["context"]
        connection_d1_d2 = format_connection(context.get("connection_d1_d2", {}))
        connection_d2_d3 = format_connection(context.get("connection_d2_d3", {}))
        
        questions_data.append({
            "answer": item["A"],          # A → answer
            "question": question_lvl_2,   # level2_question → question
            "label": data_r_mapping[idx],  # 列表格式，不换行
            "meta_data": {
                "connection_d1_d2": connection_d1_d2,
                "connection_d2_d3": connection_d2_d3,
                "question_level_1": question_lvl_1, # 新增 level 1 问题
                "question_level_0": question_lvl_0, # 新增 level 0 问题
                "original_data_source": input_path.split('/')[-2]
            }
        })

    print(f"\nR数据处理完成：")
    print(f"- 新增R记录数：{len(reference_records)}")
    print(f"- 生成有效问题数据数：{len(questions_data)}")
    print(f"- 本次分配R编号范围：{start_r_id} ~ {current_new_r_id - 1}")
    return questions_data, reference_records

def sample_random_records(records: List[Dict[str, Any]], sample_size: int = 3) -> List[Dict[str, Any]]:
    """从记录中随机抽取N条（不足则全部返回），用于重复检测"""
    if len(records) <= sample_size:
        return records.copy()
    return random.sample(records, sample_size)

def check_duplicate_qa(qa_path: str, sample_records: List[Dict[str, Any]]) -> bool:
    """
    检查抽样的QA记录是否已存在于现有JSONL文件中
    返回：True=存在重复，False=无重复
    """
    if not os.path.exists(qa_path):
        return False  # 文件不存在，无重复
    
    # 提取抽样记录的关键比对字段（question + answer）
    sample_keys = []
    for record in sample_records:
        key = (
            record.get("question", "").strip(),
            record.get("answer", "").strip(),
        )
        sample_keys.append(key)
    
    # 读取现有QA文件，比对关键字段
    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                cleaned_line = line.strip()
                if not cleaned_line:
                    continue
                try:
                    existing = json.loads(cleaned_line)
                    existing_key = (
                        existing.get("question", "").strip(),
                        existing.get("answer", "").strip(),
                    )
                    if existing_key in sample_keys:
                        print(f"❌ 发现重复数据！现有文件第{line_num}行与抽样数据重复")
                        print(f"  - 重复question：{existing.get('question', '')[:50]}...")
                        print(f"  - 重复answer：{existing.get('answer', '')[:30]}...")
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"⚠️  读取现有QA文件时发生错误：{str(e)}，跳过重复检测")
        return False  # 检测失败时默认允许写入
    
    return False

def save_output_files(questions_data: List[Dict[str, Any]], new_reference_records: List[Dict[str, Any]],
                     q_output_path: str, r_output_path: str):
    """
    保存输出文件：
    - 问题文件：JSONL格式（追加保存，每条一行，list不换行）
    - 参考文件：JSONL格式（追加保存，每条一行）
    """
    # 保存QA文件（JSONL，list不换行）
    with open(q_output_path, 'a+', encoding='utf-8') as f:
        for record in questions_data:
            # separators=(',', ':') 确保list紧凑不换行
            json.dump(record, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    print(f"\nQA文件已追加至：{q_output_path}")
    # 统计QA文件总记录数
    qa_total = sum(1 for line in open(q_output_path, 'r', encoding='utf-8') if line.strip())
    print(f"QA文件当前总记录数：{qa_total}")

    # 保存参考文件（JSONL格式，追加模式）
    with open(r_output_path, 'a+', encoding='utf-8') as f:
        for record in new_reference_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print(f"参考文件已追加至：{r_output_path}")
    # 统计参考文件总记录数
    ref_total = get_max_existing_r_count(r_output_path)
    print(f"参考文件当前总记录数：{ref_total}")

def main():
    parser = argparse.ArgumentParser(description="JSONL数据处理：过滤无效QA+重复检测+全局唯一R编号")
    parser.add_argument("--input", required=True, help="JSONL格式原始数据文件路径")
    parser.add_argument("--output-questions", required=True, help="全局QA文件路径（JSONL格式，累积保存）")
    parser.add_argument("--output-references", required=True, help="全局参考文件路径（JSONL格式，累积保存）")
    parser.add_argument("--output-invalid", required=True, help="无效数据文件路径（JSONL格式，存储answer在question中的原始数据）")
    args = parser.parse_args()

    global input_path  # 全局变量，供process_r_reference/filter_invalid_data使用
    input_path = args.input

    try:
        # 1. 加载原始数据（基础格式/字段过滤）
        original_data = load_original_data(args.input)
        if not original_data:
            print("警告：未加载到有效数据，脚本终止")
            return
        
        # 2. 过滤无效数据（answer完整出现在question中）+ 分离无效数据
        print("\n==================================================")
        print("🔍 开始过滤无效数据（answer完整出现在question中）...")
        print("==================================================")
        filtered_data, invalid_data = filter_invalid_data(original_data)
        if not filtered_data:
            print("警告：过滤后无有效数据，仅保存无效数据")
            # 保存无效数据后终止
            save_invalid_data(invalid_data, args.output_invalid)
            return
        
        # 3. 保存无效数据（无论是否有有效数据，都要保存）
        save_invalid_data(invalid_data, args.output_invalid)
        
        # 4. 获取参考文件现有R记录数，确定本次起始编号
        existing_count = get_max_existing_r_count(args.output_references)
        start_r_id = existing_count + 1
        
        # 5. 处理R数据和编号分配（仅处理过滤后的有效数据）
        questions_data, new_reference_records = process_r_reference(filtered_data, start_r_id)
        if not questions_data:
            print("警告：未生成有效QA数据，脚本终止")
            return
        
        # 6. 重复检测：随机抽3条比对现有QA文件
        print("\n==================================================")
        print("🔍 开始重复检测...")
        print("==================================================")
        sample_records = sample_random_records(questions_data)
        is_duplicate = check_duplicate_qa(args.output_questions, sample_records)
        if is_duplicate:
            print("❌ 错误：当前文件数据已存在于QA文件中，已跳过保存！")
            return
        
        # 7. 保存输出文件（QA+参考文件）
        print("\n==================================================")
        print("💾 开始保存有效文件...")
        print("==================================================")
        save_output_files(questions_data, new_reference_records, args.output_questions, args.output_references)
        
        print("\n✅ 处理完全完成！")
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()