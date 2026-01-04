# -*- coding: utf-8 -*-
"""
知识图谱构建与推理路径生成工具 (语义增强版 V3.3 - 强力消歧义版 + 全局唯一性校验)

新增改进：
1. [Global Unique] 接入全局 doc_id 列表基准文件（unique_doc_id_lists.jsonl）；
2. [Fix] 修复原指纹丢失路径顺序的问题（改用列表转字符串作为指纹）；
3. [Global] 新生成的路径需同时满足“本次唯一”+“全局唯一”；
4. [Persist] 可选：生成新路径后更新全局基准文件（保证后续运行也不重复）。
"""

import os
import glob
import pandas as pd
import networkx as nx
import random
import json
import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Set, Tuple
from itertools import combinations
from tqdm import tqdm
from rapidfuzz import fuzz
from functools import partial

# --- 全局常量: 增强版黑名单 ---
BRIDGE_BLOCKLIST = {
    # --- 通用章节/概念词 ---
    "history", "music", "love", "time", "life", "career", "background", "overview",
    "references", "external links", "see also", "early life", "personal life",
    "education", "death", "family", "awards", "works", "filmography",
    "discography", "bibliography", "notes", "further reading", "introduction",
    "production", "reception", "legacy", "style", "influence", "summary",
    "series", "season", "episode", "cast", "plot", "synopsis", "contents",
    "track listing", "personnel", "charts", "certifications", "release history",
    "critical reception", "accolades", "box office",
    
    # --- 高频歧义单词 ---
    "stay", "deception", "truth", "dare", "home", "run", "live", "believe",
    "university", "college", "school", "high school", "academy", "institute",
    "town", "city", "village", "county", "state", "province", "region", "country",
    "island", "mountain", "river", "lake", "sea", "ocean", "park", "valley",
    "north", "south", "east", "west", "central", "upper", "lower",
    "first", "second", "third", "new", "old", "great", "best", "top",
    "unknown", "various", "none", "list", "category", "type", "part", "member",
    "born", "died", "known", "located", "founded", "established",
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december",
    "album", "song", "single", "band", "group", "artist", "record",
    "film", "movie", "show", "book", "novel", "play", "game", "character",
    "king", "queen", "prince", "princess", "president", "governor", "emperor",
    "father", "mother", "son", "daughter", "brother", "sister", "wife", "husband",
    "god", "church", "saint", "st.", "holy", "divine", "mythology",
    "world", "earth", "sun", "moon", "star", "space", "universe",
    "war", "battle", "army", "navy", "force", "peace", "commander",
    "english", "french", "german", "spanish", "italian", "japanese", "chinese"
}

SPECIFIC_ENTITY_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "with", "as", "by", "at", "from", "to", "and", "or"}
SPACY_MODEL_NAME = "en_core_web_trf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
VALID_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT", "EVENT", "FAC", "LAW", "NORP"}
SIMILARITY_THRESHOLD = 0.30

# --- 新增：全局唯一性校验相关常量 ---
GLOBAL_UNIQUE_DOC_IDS_PATH = "/mnt/cxzx/share/xiaojunqian/data/unique_doc_id_lists.jsonl"  # 全局基准文件路径
UPDATE_GLOBAL_FILE = True  # 是否在生成新路径后更新全局基准文件

# --- 全局变量 ---
nlp = None
embedding_model = None
NUM_ENTITIES_TO_EXTRACT_GLOBAL = 50
PROCESSED_TITLE_MAP_GLOBAL = None
TARGET_EMBEDDINGS_MAP = None

# --- 函数定义 ---

def load_and_prepare_data(path: str, start_index: int, num_files: int, num_samples: int) -> pd.DataFrame:
    """加载和预处理数据"""
    print(f"[*] 1/5: 正在从 {path} 加载数据...")
    parquet_files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"在路径 {path} 中没有找到 Parquet 文件。")

    if start_index >= len(parquet_files):
        raise IndexError(f"起始索引 {start_index} 超出范围。")

    if num_files == -1:
        files_to_process = parquet_files[start_index:]
    else:
        end_index = start_index + num_files
        files_to_process = parquet_files[start_index:end_index]

    if not files_to_process:
         raise ValueError(f"未选中任何文件。")

    print(f"    - 将处理 {len(files_to_process)} 个文件。")
    df_list = [pd.read_parquet(file) for file in tqdm(files_to_process, desc="    - 读取文件")]
    if not df_list:
        raise ValueError("未能成功加载任何数据。")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.dropna(subset=['id', 'title', 'text'], inplace=True)
    combined_df = combined_df.reset_index(drop=True)
    print(f"    - 合并后共有 {len(combined_df)} 篇文章。")

    if num_samples > 0 and num_samples < len(combined_df):
        print(f"    - 随机采样 {num_samples} 篇...")
        combined_df = combined_df.sample(n=num_samples, random_state=42).copy()

    return combined_df

def worker_init_per_gpu(num_entities_to_extract: int, title_map: Dict[str, str]):
    """Worker 初始化：加载 Spacy 和 Embedding 模型。"""
    global nlp, embedding_model, NUM_ENTITIES_TO_EXTRACT_GLOBAL, PROCESSED_TITLE_MAP_GLOBAL, TARGET_EMBEDDINGS_MAP
    
    NUM_ENTITIES_TO_EXTRACT_GLOBAL = num_entities_to_extract
    PROCESSED_TITLE_MAP_GLOBAL = title_map
    
    import spacy
    import torch
    from sentence_transformers import SentenceTransformer
    
    try:
        p = mp.current_process()
        if hasattr(p, '_identity') and p._identity:
            worker_id = p._identity[0] - 1
        else:
            worker_id = random.randint(0, 7)
            
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
        num_gpus = len(visible_devices)
        gpu_index = worker_id % num_gpus
        device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
        
        # 1. 加载 Spacy (包含 sentencizer 修复)
        try:
            if device.startswith('cuda'): spacy.prefer_gpu(gpu_index)
            nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "lemmatizer"])
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        except Exception as e:
            print(f"    - [PID {os.getpid()}] Spacy GPU 失败: {e}，切换 CPU。")
            nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "lemmatizer"])
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

        # 2. 加载 Embedding 模型
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            
            # 3. 预计算目标标题向量
            target_titles = list(title_map.keys())
            if target_titles:
                title_embeddings = embedding_model.encode(
                    target_titles, 
                    batch_size=4096, 
                    convert_to_tensor=True, 
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                TARGET_EMBEDDINGS_MAP = {}
                for i, title in enumerate(target_titles):
                    TARGET_EMBEDDINGS_MAP[title] = title_embeddings[i]
            else:
                TARGET_EMBEDDINGS_MAP = {}
                
        except Exception as e:
            print(f"    - [PID {os.getpid()}] Embedding 初始化失败: {e}")
            embedding_model = None
            
    except Exception as e:
        print(f"    - [PID {os.getpid()}] 初始化严重错误: {e}")

def preprocess_text_for_matching(text: str) -> str:
    """预处理文本"""
    if not isinstance(text, str): return ""
    text = text.strip().lower()
    if text.startswith("the "): return text[4:]
    elif text.startswith("a "): return text[2:]
    elif text.startswith("an "): return text[3:]
    return text

def extract_specific_entities(text: str) -> Set[str]:
    """从文本中提取实体 (包含强力过滤逻辑)"""
    global nlp, NUM_ENTITIES_TO_EXTRACT_GLOBAL
    if nlp is None or not isinstance(text, str) or not text: return set()
    
    doc = nlp(text[:300_000])
    
    specific_entities = set()
    for ent in doc.ents:
        entity_text = ent.text.strip()
        entity_lower = entity_text.lower()
        
        # --- 基础规则 ---
        if len(entity_text) < 3: continue
        if not entity_text[0].isupper(): continue
        
        # --- 黑名单 ---
        first_word = entity_lower.split()[0]
        if first_word in SPECIFIC_ENTITY_STOPWORDS: continue
        if entity_lower in BRIDGE_BLOCKLIST: continue
        
        # --- 类型检查 ---
        if ent.label_ not in VALID_ENTITY_LABELS: continue
        
        # --- 单字过滤策略 ---
        ALLOWED_SINGLE_WORD_LABELS = {"GPE", "ORG", "NORP", "LOC"}
        if ent.label_ not in ALLOWED_SINGLE_WORD_LABELS and len(entity_text.split()) < 2:
            continue
            
        # --- 排除伪实体 ---
        if ent.label_ in {"ORG", "WORK_OF_ART"} and entity_lower in {"band", "song", "album", "book", "movie", "show", "series"}:
            continue

        specific_entities.add(entity_text)
            
    if len(specific_entities) > NUM_ENTITIES_TO_EXTRACT_GLOBAL:
        return set(random.sample(list(specific_entities), NUM_ENTITIES_TO_EXTRACT_GLOBAL))
    return specific_entities

def find_edges_for_doc(args: Tuple) -> List[Tuple]:
    """查找边 (包含 标题+上下文 的联合语义验证)"""
    from sentence_transformers import util
    
    source_id, source_text, source_title = args
    global PROCESSED_TITLE_MAP_GLOBAL, embedding_model, TARGET_EMBEDDINGS_MAP, nlp
    
    potential_edges = []
    
    if nlp is None: return []
    
    # 截断文本
    doc = nlp(source_text[:300_000])
    
    extracted_candidates = []
    
    for ent in doc.ents:
        entity_text = ent.text.strip()
        entity_lower = entity_text.lower()
        
        # --- 复制过滤逻辑 ---
        if len(entity_text) < 3: continue
        if not entity_text[0].isupper(): continue
        
        first_word = entity_lower.split()[0]
        if first_word in SPECIFIC_ENTITY_STOPWORDS: continue
        if entity_lower in BRIDGE_BLOCKLIST: continue
        if ent.label_ not in VALID_ENTITY_LABELS: continue
        
        ALLOWED_SINGLE_WORD_LABELS = {"GPE", "ORG", "NORP", "LOC"}
        if ent.label_ not in ALLOWED_SINGLE_WORD_LABELS and len(entity_text.split()) < 2:
            continue

        if ent.label_ in {"ORG", "WORK_OF_ART"} and entity_lower in {"band", "song", "album", "group"}: continue
        
        # 获取上下文 (双重保险 + 标题增强)
        try:
            raw_context = ent.sent.text
            context = f"{source_title}. {raw_context}" 
        except Exception:
            start = max(0, ent.start_char - 100)
            end = min(len(source_text), ent.end_char + 100)
            raw_context = source_text[start:end]
            context = f"{source_title}. {raw_context}"

        if len(raw_context) < 10: continue
        extracted_candidates.append((entity_text, context))

    if len(extracted_candidates) > NUM_ENTITIES_TO_EXTRACT_GLOBAL:
        extracted_candidates = random.sample(extracted_candidates, NUM_ENTITIES_TO_EXTRACT_GLOBAL)

    # 2. 匹配与语义验证
    for entity_text, source_context in extracted_candidates:
        processed_entity = preprocess_text_for_matching(entity_text)
        
        if processed_entity in PROCESSED_TITLE_MAP_GLOBAL:
            target_id = PROCESSED_TITLE_MAP_GLOBAL[processed_entity]
            if source_id == target_id: continue
            
            sim_score = 1.0
            
            # Embedding 语义验证
            if embedding_model is not None and TARGET_EMBEDDINGS_MAP is not None:
                try:
                    if processed_entity in TARGET_EMBEDDINGS_MAP:
                        target_emb = TARGET_EMBEDDINGS_MAP[processed_entity]
                        
                        source_emb = embedding_model.encode(source_context, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
                        sim_score = util.cos_sim(source_emb, target_emb).item()
                        
                        if sim_score < SIMILARITY_THRESHOLD:
                            continue 
                except Exception:
                    pass

            edge_data = {'bridge': entity_text, 'edit_distance': 0, 'similarity': sim_score}
            potential_edges.append((source_id, target_id, edge_data))
            
    return potential_edges

def build_entity_title_graph(df: pd.DataFrame, processes_per_gpu: int, max_in_degree: int, num_entities_to_extract: int) -> nx.DiGraph:
    """并行构建知识图谱"""
    print(f"[*] 2/5: 正在通过“精准匹配 + 语义验证”构建知识图谱...")
    
    processed_title_map = {
        preprocess_text_for_matching(row['title']): row['id']
        for _, row in df.iterrows()
    }
    
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    num_processes = num_gpus * processes_per_gpu
    
    print(f"    - GPU 进程数: {num_processes}")

    tasks = [(row.id, row.text, row.title) for row in df.itertuples(index=False)]
    all_potential_edges = []
    
    init_func = partial(worker_init_per_gpu, num_entities_to_extract, processed_title_map)
    chunk_size = 20 

    with mp.Pool(processes=num_processes, initializer=init_func) as pool:
        with tqdm(total=len(tasks), desc="    - 并行处理") as pbar:
            for result_edges in pool.imap_unordered(find_edges_for_doc, tasks, chunksize=chunk_size):
                if result_edges:
                    all_potential_edges.extend(result_edges)
                pbar.update(1)

    print(f"    - 发现 {len(all_potential_edges)} 条潜在连接。")
    print(f"    - 阶段 2/2: 构建图谱 (入度限制 {max_in_degree})...")
    
    G = nx.DiGraph()
    G.add_nodes_from(df['id'])
    random.shuffle(all_potential_edges)
    
    bridge_in_degree_counts = {}
    for source_id, target_id, edge_data in tqdm(all_potential_edges):
        bridge_entity = edge_data['bridge']
        counter_key = (target_id, bridge_entity.lower())
        current_count = bridge_in_degree_counts.get(counter_key, 0)
        if current_count < max_in_degree:
            G.add_edge(source_id, target_id, **edge_data)
            bridge_in_degree_counts[counter_key] = current_count + 1
            
    print(f"    - 节点数: {G.number_of_nodes()} | 边数: {G.number_of_edges()}")
    return G

# --- 新增：加载全局唯一的 doc_id 列表 ---
def load_global_unique_doc_id_lists(global_path: str) -> Set[str]:
    """
    加载全局基准文件中的 doc_id 列表，转为字符串指纹集合（保证顺序+内容唯一）
    """
    global_fingerprints = set()
    if not os.path.exists(global_path):
        print(f"[Warning] 全局基准文件不存在: {global_path}，将创建新文件。")
        return global_fingerprints
    
    print(f"[*] 加载全局唯一 doc_id 列表: {global_path}")
    with open(global_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="    - 读取全局列表"):
            try:
                item = json.loads(line.strip())
                doc_id_list = item.get('document_ids', [])
                # 转为字符串指纹（保留顺序）
                list_fingerprint = json.dumps(doc_id_list, sort_keys=False)
                global_fingerprints.add(list_fingerprint)
            except json.JSONDecodeError:
                print(f"[Warning] 全局文件行格式错误，跳过: {line[:100]}")
                continue
    
    print(f"    - 加载到 {len(global_fingerprints)} 个全局唯一的 doc_id 列表")
    return global_fingerprints

# --- 新增：更新全局基准文件 ---
def update_global_doc_id_list(global_path: str, new_doc_id_lists: List[List[str]]):
    """
    将新生成的 doc_id 列表追加到全局基准文件
    """
    if not UPDATE_GLOBAL_FILE or not new_doc_id_lists:
        return
    
    print(f"[*] 更新全局基准文件: {global_path}")
    # 先读取现有内容（避免重复写入）
    existing_fingerprints = load_global_unique_doc_id_lists(global_path)
    
    with open(global_path, 'a', encoding='utf-8') as f:
        added_count = 0
        for doc_id_list in new_doc_id_lists:
            list_fingerprint = json.dumps(doc_id_list, sort_keys=False)
            if list_fingerprint not in existing_fingerprints:
                f.write(json.dumps({"document_ids": doc_id_list}, ensure_ascii=False) + '\n')
                added_count += 1
                existing_fingerprints.add(list_fingerprint)
    
    print(f"    - 向全局文件新增 {added_count} 个唯一的 doc_id 列表")

def create_reasoning_paths_dataset(
    graph: nx.DiGraph, 
    id_to_doc_map: Dict, 
    num_paths: int, 
    num_hops: int, 
    similarity_threshold: float,
    global_unique_fingerprints: Set[str]  # 新增：全局指纹集合
) -> List[Dict]:
    print(f"[*] 3/5: 正在生成 {num_paths} 条唯一的 {num_hops}-跳 推理路径 (全局唯一版)...")
    if graph.number_of_edges() == 0:
        return []
        
    dataset = []
    # 1. 本次运行的临时指纹集合（避免本次重复）
    seen_paths_fingerprints = set() 
    # 2. 存储新生成的 doc_id 列表（用于更新全局文件）
    new_generated_doc_id_lists = []
    
    nodes_with_out_edges = [node for node, out_degree in graph.out_degree() if out_degree > 0]
    if not nodes_with_out_edges: return []
        
    with tqdm(total=num_paths, desc="    - 生成路径") as pbar:
        attempts = 0
        max_attempts = num_paths * 200 
        while len(dataset) < num_paths and attempts < max_attempts:
            attempts += 1
            start_node = random.choice(nodes_with_out_edges)
            path = generate_fixed_hop_path(graph, start_node, num_hops)
            
            # 跳过长度不符合的路径
            if not path or len(path) != num_hops + 1: continue
            
            # --- 步骤1：提取路径的 doc_id 列表，生成指纹（保留顺序）---
            doc_id_list = path  # path 本身就是 doc_id 列表
            path_fingerprint = json.dumps(doc_id_list, sort_keys=False)  # 修复：保留顺序的指纹
            
            # --- 步骤2：全局唯一性校验 ---
            if path_fingerprint in global_unique_fingerprints:
                continue  # 全局已存在，跳过
            
            # --- 步骤3：本次运行唯一性校验 ---
            if path_fingerprint in seen_paths_fingerprints:
                continue  # 本次已生成，跳过
            
            # --- 步骤4：路径有效性验证（原逻辑保留）---
            if not is_path_valid(path, graph, similarity_threshold):
                continue
            
            # --- 步骤5：通过所有校验，确定路径并保存 ---
            seen_paths_fingerprints.add(path_fingerprint)
            global_unique_fingerprints.add(path_fingerprint)  # 加入全局集合，避免后续重复
            formatted_path = package_path_to_custom_format(path, graph, id_to_doc_map)
            dataset.append(formatted_path)
            new_generated_doc_id_lists.append(doc_id_list)  # 记录新生成的 doc_id 列表
            pbar.update(1)
    
    # 可选：更新全局基准文件
    update_global_doc_id_list(GLOBAL_UNIQUE_DOC_IDS_PATH, new_generated_doc_id_lists)
    
    return dataset

def generate_fixed_hop_path(graph: nx.DiGraph, start_node: str, num_hops: int) -> List[str]:
    path = [start_node]
    current_node = start_node
    for _ in range(num_hops):
        try:
            neighbors = [n for n in list(graph.successors(current_node)) if n not in path]
        except: return None
        if not neighbors: return None 
        next_node = random.choice(neighbors)
        path.append(next_node)
        current_node = next_node
    return path

def package_path_to_custom_format(path: List[str], graph: nx.DiGraph, id_to_doc_map: Dict) -> Dict:
    output_item = {}
    for i, doc_id in enumerate(path):
        output_item[f'document{i+1}'] = id_to_doc_map[doc_id]
        if i < len(path) - 1:
            next_doc_id = path[i+1]
            edge_data = graph.get_edge_data(doc_id, next_doc_id)
            output_item[f'connection_d{i+1}_d{i+2}'] = edge_data if edge_data else {'bridge': 'UNKNOWN'}
    return output_item

def is_path_valid(path: List[str], graph: nx.DiGraph, similarity_threshold: float) -> bool:
    if len(path) < 3: return True
    try:
        bridges = [graph.get_edge_data(path[i], path[i+1])['bridge'] for i in range(len(path) - 1)]
    except: return False
    if len(bridges) < 2: return True
    for bridge1, bridge2 in combinations(bridges, 2):
        if fuzz.ratio(bridge1, bridge2) > similarity_threshold: return False
    return True

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1. 加载数据
    wiki_df = load_and_prepare_data(args.wiki_data_path, args.start_file_index, args.num_files, args.num_samples)
    id_to_doc_map = {row['id']: {'id': row['id'], 'title': row['title'], 'text': row['text']} for _, row in wiki_df.iterrows()}
    
    # 2. 加载/构建图谱
    graph_file = os.path.join(args.output_path, "knowledge_graph.gml")
    if os.path.exists(graph_file) and not args.force_rebuild_graph:
        print(f"[*] 2/5: 从文件加载图...")
        knowledge_graph = nx.read_gml(graph_file)
    else:
        knowledge_graph = build_entity_title_graph(
            wiki_df, args.processes_per_gpu, args.max_in_degree_per_bridge, args.num_entities_to_extract
        )
        print(f"    - 保存图...")
        nx.write_gml(knowledge_graph, graph_file)

    # --- 新增：加载全局唯一 doc_id 列表 ---
    global_unique_fingerprints = load_global_unique_doc_id_lists(GLOBAL_UNIQUE_DOC_IDS_PATH)
    
    # 3. 生成全局唯一的推理路径
    reasoning_paths_dataset = create_reasoning_paths_dataset(
        knowledge_graph, 
        id_to_doc_map, 
        args.num_paths_to_generate, 
        args.num_hops, 
        args.bridge_similarity_threshold,
        global_unique_fingerprints  # 传入全局指纹集合
    )

    if reasoning_paths_dataset:
        dataset_file = os.path.join(args.output_path, f"unique_{args.num_hops+1}R_{args.num_hops}hops.jsonl")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in reasoning_paths_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"[*] 完成！生成了 {len(reasoning_paths_dataset)} 条全局唯一的路径。")
    else:
        print("[*] 任务结束，未生成路径。")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--start_file_index", type=int, default=0)
    parser.add_argument("--num_files", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--num_entities_to_extract", type=int, default=50)
    parser.add_argument("--force_rebuild_graph", action='store_true')
    parser.add_argument("--max_in_degree_per_bridge", type=int, default=30)
    parser.add_argument("--num_paths_to_generate", type=int, default=100000)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--bridge_similarity_threshold", type=float, default=60.0)
    parser.add_argument("--gpus", type=str, default="4,5,6,7")
    parser.add_argument("--processes_per_gpu", type=int, default=4) 
    return parser.parse_args()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main(parse_args())