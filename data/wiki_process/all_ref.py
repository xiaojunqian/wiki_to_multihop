import os
import json
import argparse
import uuid
import glob
import re
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 环境配置 (按要求指定显卡)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# ==========================================
# 2. 辅助函数: 段落拆分规则
# ==========================================
def split_into_paragraphs(text: str):
    """
    [Splitting Rule]
    使用正则表达式将文本分割成段落，并清理每个段落。
    规则: 在句子结束符(.!?)后紧跟换行符的位置进行切分。
    """
    if not text:
        return []
    
    # 使用正向后行断言在句子结束符和换行符后分割
    # 逻辑：找到 [.!?] 后面紧跟换行符的地方进行切分
    paragraphs = re.split(r'(?<=[.!?])\s*\n+', text.strip())
    
    # 清理每个段落，移除内部换行符并去除首尾空格
    return [p.strip().replace('\n', ' ') for p in paragraphs if p.strip()]

# ==========================================
# 3. 核心逻辑: 拆分 -> 拼接 -> 截断 -> 过滤
# ==========================================
def apply_strict_processing_logic(raw_content):
    """
    [Core Logic]
    复刻 process_single_item_task 中的拼接与截断逻辑。
    """
    final_content = ""
    type_tag = ""
    content_list = []

    # --- 步骤 A: 统一转为列表 (List of chunks) ---
    if isinstance(raw_content, str):
        # 如果是字符串 (Wiki全文)，先应用拆分规则
        content_list = split_into_paragraphs(raw_content)
        type_tag = "split_from_str"
    elif isinstance(raw_content, list):
        # 如果已经是列表，直接使用
        content_list = raw_content
        type_tag = "original_list"
    else:
        # 其他情况强转列表
        content_list = [str(raw_content)]
        type_tag = "fallback_list"

    # --- 步骤 B: 严格拼接逻辑 (Strict Splicing) ---
    if len(content_list) > 0:
        try:
            if len(content_list) == 1:
                final_content = content_list[0]
            else:
                # === 核心循环 ===
                temp_i = 1
                tempstr = content_list[0]
                
                # [关键规则] 只有当前长度 < 200 时，才继续拼接下一段
                while temp_i < len(content_list) and len(tempstr) < 200:
                    # 原逻辑: tempstr += item[r_key][temp_i]
                    tempstr += content_list[temp_i]
                    temp_i += 1
                
                final_content = tempstr
        except Exception:
            # 异常回退
            final_content = str(raw_content)[:500]
            type_tag += "_error"
    else:
        # 空列表情况
        return None, False, "empty_list"

    # --- 步骤 C: 歧义页检查 (已补回) ---
    # 对应原逻辑: if "may refer to:" in content: return {"error": ...}
    if "may refer to:" in final_content:
        return None, False, "disambiguation_page_filtered"

    # --- 步骤 D: 空内容检查 ---
    if not final_content.strip():
        return None, False, "empty_content"

    return final_content, True, type_tag

# ==========================================
# 4. 主流程: 遍历 Parquet -> 生成 Ref
# ==========================================
def process_wiki_data(wiki_data_path, output_file):
    print(f"[*] 显卡环境: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[*] Wiki 数据路径: {wiki_data_path}")
    print(f"[*] 输出文件: {output_file}")

    # 1. 扫描 Parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(wiki_data_path, "*.parquet")))
    if not parquet_files:
        print(f"❌ 在 {wiki_data_path} 未找到 .parquet 文件，尝试直接读取...")
        parquet_files = sorted(glob.glob(wiki_data_path)) 
    
    if not parquet_files:
        raise FileNotFoundError("未找到任何输入文件。")

    print(f"[*] 共发现 {len(parquet_files)} 个文件。")

    # 2. 准备输出
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 全局去重集合 (基于 Wiki ID)
    seen_doc_ids = set()
    
    total_records = 0
    valid_refs = 0
    duplicate_refs = 0
    skipped_refs = 0

    # 3. 逐文件处理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in tqdm(parquet_files, desc="Processing Parquet Files"):
            try:
                # 使用 PyArrow 引擎读取
                df = pd.read_parquet(file_path, engine='auto')
                
                # 检查必要列
                required_cols = ['id', 'text']
                if not all(col in df.columns for col in required_cols):
                    print(f"⚠️ 文件 {os.path.basename(file_path)} 缺少必要列 (id, text)，跳过。")
                    continue
                
                records = df.to_dict('records')
                
                for row in records:
                    total_records += 1
                    
                    # --- A. ID 获取与去重 ---
                    doc_id = str(row['id']).strip()
                    if doc_id in seen_doc_ids:
                        duplicate_refs += 1
                        continue
                    
                    # --- B. 数据提取 ---
                    raw_text = row['text']
                    title = row.get('title', '').strip()
                    
                    # --- C. 应用拆分+拼接+过滤逻辑 ---
                    final_content, is_valid, tag = apply_strict_processing_logic(raw_text)
                    
                    if not is_valid:
                        skipped_refs += 1
                        continue
                    
                    # --- D. 构造 Ref 对象 ---
                    # [Format] Title + \n + Content (Spliced < 200 chars)
                    final_value = f"{title}\n{final_content}" if title else final_content
                    
                    ref_item = {
                        "id": str(uuid.uuid4()),
                        "value": final_value,
                        "meta_data": {
                            "document_id": doc_id,
                            "title": title,
                            "original_source": "wiki_dump",
                            "process_type": tag, 
                            "is_truncated": True 
                        }
                    }
                    
                    # --- E. 写入与记录 ---
                    f_out.write(json.dumps(ref_item, ensure_ascii=False) + '\n')
                    seen_doc_ids.add(doc_id)
                    valid_refs += 1
                    
            except Exception as e:
                print(f"❌ 处理文件 {file_path} 时出错: {e}")

    print("\n" + "="*40)
    print("🎉 处理完成 (Processing Complete)")
    print(f"📊 总扫描文档: {total_records}")
    print(f"✅ 生成 Ref 数: {valid_refs}")
    print(f"⏭️  全局去重数: {duplicate_refs}")
    print(f"🗑️  无效/歧义数: {skipped_refs}")
    print(f"📂 结果保存至: {output_file}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wiki Ref 生成器 (拆分+严格拼接+消歧义版)")
    
    parser.add_argument("--wiki_data_path", type=str, required=True, 
                        help="包含 .parquet 文件的目录路径")
    parser.add_argument("--output", type=str, required=True, 
                        help="输出 .jsonl 文件的完整路径")
    
    args = parser.parse_args()
    
    process_wiki_data(args.wiki_data_path, args.output)