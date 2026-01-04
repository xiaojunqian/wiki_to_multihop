# -*- coding: utf-8 -*-
"""
知识图谱构建与推理路径生成工具 (语义增强版 V3.3 - 强力消歧义版)

改进日志：
1. [Logic] 引入 Source Title 增强 Embedding：将“源文档标题”加入上下文。
2. [Strategy] **全面升级单字封杀令**：
   - 之前仅过滤 PERSON。
   - 现在扩展到 WORK_OF_ART (作品), PRODUCT (产品), EVENT (事件), FAC (设施), LAW (法律)。
   - 仅保留 GPE (地名), ORG (组织), NORP (民族) 的单字实体（如 "China", "Google", "Japanese"），因为这些通常具有唯一性。
   - 解决潜在的 Spacy 误分类问题（如把人名误认为组织）。
3. [Strategy] 语义阈值优化：保持 0.30。
4. [Fix] 修复 Spacy 崩溃问题：保持 sentencizer。
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
# 请确保已下载模型或网络通畅，建议使用本地绝对路径
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
VALID_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT", "EVENT", "FAC", "LAW", "NORP"}

# [Strategy] 阈值设定
SIMILARITY_THRESHOLD = 0.30

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
    """
    Worker 初始化：加载 Spacy 和 Embedding 模型。
    """
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
            # [FIX] 添加 sentencizer 防止 ent.sent 报错
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
            
            # 3. 预计算目标标题向量 (利用显存加速后续查询)
            target_titles = list(title_map.keys())
            if target_titles:
                # 批量编码
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
    """
    从文本中提取实体 (包含强力过滤逻辑)
    """
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
        
        # --- [CRITICAL UPDATE] 全面升级单字过滤策略 ---
        # 除了 GPE (地名), ORG (组织), NORP (民族) 之外，其他类型实体如果是单字，一律丢弃。
        # 这能防止 "Valli" (Person/Org), "Stay" (Work of Art), "War" (Event) 等歧义。
        # 保留 ORG 是因为像 "Google", "Apple" 往往是单字但明确。
        # 保留 GPE 是因为像 "London", "Japan" 是单字且明确。
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
    """
    查找边 (包含 标题+上下文 的联合语义验证)
    """
    from sentence_transformers import util
    
    # [NEW] 接收 title 参数
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
        
        # [CRITICAL UPDATE] 全面升级单字过滤策略 (与 extract_specific_entities 保持一致)
        ALLOWED_SINGLE_WORD_LABELS = {"GPE", "ORG", "NORP", "LOC"}
        if ent.label_ not in ALLOWED_SINGLE_WORD_LABELS and len(entity_text.split()) < 2:
            continue

        if ent.label_ in {"ORG", "WORK_OF_ART"} and entity_lower in {"band", "song", "album", "group"}: continue
        
        # 获取上下文 (双重保险 + 标题增强)
        try:
            raw_context = ent.sent.text
            # [Strategy] 标题增强上下文：明确"谁"在做这件事
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
                        
                        # 计算源端向量
                        source_emb = embedding_model.encode(source_context, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
                        sim_score = util.cos_sim(source_emb, target_emb).item()
                        
                        # [Strategy] 0.30 阈值
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

    # [NEW] 传递 title 字段
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

def create_reasoning_paths_dataset(graph: nx.DiGraph, id_to_doc_map: Dict, num_paths: int, num_hops: int, similarity_threshold: float) -> List[Dict]:
    print(f"[*] 3/5: 正在生成 {num_paths} 条唯一的 {num_hops}-跳 推理路径...")
    if graph.number_of_edges() == 0:
        return []
        
    dataset = []
    seen_paths_fingerprints = set() 
    nodes_with_out_edges = [node for node, out_degree in graph.out_degree() if out_degree > 0]
    if not nodes_with_out_edges: return []
        
    with tqdm(total=num_paths, desc="    - 生成路径") as pbar:
        attempts = 0
        max_attempts = num_paths * 200 
        while len(dataset) < num_paths and attempts < max_attempts:
            attempts += 1
            start_node = random.choice(nodes_with_out_edges)
            path = generate_fixed_hop_path(graph, start_node, num_hops)
            if not path or len(path) != num_hops + 1: continue
            try:
                bridges = [graph.get_edge_data(path[i], path[i+1])['bridge'].lower() for i in range(len(path) - 1)]
            except: continue
            path_fingerprint = (frozenset(path), frozenset(bridges))
            if path_fingerprint in seen_paths_fingerprints: continue
            if not is_path_valid(path, graph, similarity_threshold): continue
            seen_paths_fingerprints.add(path_fingerprint)
            formatted_path = package_path_to_custom_format(path, graph, id_to_doc_map)
            dataset.append(formatted_path)
            pbar.update(1)
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
    wiki_df = load_and_prepare_data(args.wiki_data_path, args.start_file_index, args.num_files, args.num_samples)
    id_to_doc_map = {row['id']: {'id': row['id'], 'title': row['title'], 'text': row['text']} for _, row in wiki_df.iterrows()}
    
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

    reasoning_paths_dataset = create_reasoning_paths_dataset(
        knowledge_graph, id_to_doc_map, args.num_paths_to_generate, args.num_hops, args.bridge_similarity_threshold
    )

    if reasoning_paths_dataset:
        dataset_file = os.path.join(args.output_path, f"unique_{args.num_hops+1}R_{args.num_hops}hops.jsonl")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in reasoning_paths_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"[*] 完成！生成了 {len(reasoning_paths_dataset)} 条路径。")
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