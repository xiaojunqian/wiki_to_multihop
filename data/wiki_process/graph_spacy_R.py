# -*- coding: utf-8 -*-
"""
知识图谱构建与推理路径生成工具

本脚本通过以下步骤从维基百科数据中生成多跳推理路径数据集：
1. 从Parquet格式的维基百科转储文件中加载和预处理文章。
2. 使用多GPU和多进程，通过Spacy进行命名实体识别（NER）。
3. 基于“实体-标题”的精准匹配，构建一个有向知识图谱，其中节点是文章，边由识别出的实体（桥梁）连接。
4. 从图中生成并采样指定跳数的推理路径，同时确保路径的唯一性（基于文章和桥梁的组合）和多样性（桥梁之间不太相似）。
5. 将生成的路径数据集保存为JSONL格式。
"""

import os
import glob
import pandas as pd
import networkx as nx
import random
import json
import argparse
import multiprocessing as mp
from typing import List, Dict, Set, Tuple
from itertools import combinations
from tqdm import tqdm
from rapidfuzz import fuzz

# --- 全局常量 (不轻易变动的配置) ---
SPECIFIC_ENTITY_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "with", "as", "by", "at"}
SPACY_MODEL_NAME = "en_core_web_trf"
VALID_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT", "EVENT", "FAC", "LAW", "NORP"}

# --- 全局变量 (由工作进程初始化) ---
# 这个变量在每个子进程中是独立的，通过 worker_init_per_gpu 进行初始化。
nlp = None

# --- 函数定义 ---

def load_and_prepare_data(path: str, start_index: int, num_files: int, num_samples: int) -> pd.DataFrame:
    """
    从指定路径加载、切片和采样Parquet文件。

    Args:
        path (str): 存放Parquet文件的目录路径。
        start_index (int): 文件列表的起始读取索引。
        num_files (int): 从起始索引开始要读取的文件数量 (-1 表示读取到末尾)。
        num_samples (int): 从加载的所有数据中随机采样的文章数量 (-1 表示使用全部数据)。

    Returns:
        pd.DataFrame: 经过处理和采样的DataFrame。

    Raises:
        FileNotFoundError: 如果在指定路径下找不到任何Parquet文件。
        IndexError: 如果起始索引超出文件列表范围。
        ValueError: 如果没有加载到任何数据或根据配置没有选中任何文件。
    """
    print(f"[*] 1/5: 正在从 {path} 加载数据...")
    parquet_files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"在路径 {path} 中没有找到 Parquet 文件。")

    total_files_found = len(parquet_files)
    print(f"    - 总共发现 {total_files_found} 个Parquet文件。")

    if start_index >= total_files_found:
        raise IndexError(f"错误：起始索引 {start_index} 超出范围。只找到了 {total_files_found} 个文件。")

    if num_files == -1:
        files_to_process = parquet_files[start_index:]
    else:
        end_index = start_index + num_files
        files_to_process = parquet_files[start_index:end_index]

    if not files_to_process:
         raise ValueError(f"根据设置 (start={start_index}, num={num_files})，没有选中任何文件进行处理。")

    print(f"    - 将处理 {len(files_to_process)} 个文件 (索引从 {start_index} 到 {start_index + len(files_to_process) - 1})。")

    df_list = [pd.read_parquet(file) for file in tqdm(files_to_process, desc="    - 正在读取文件")]
    if not df_list:
        raise ValueError("未能成功加载任何 Parquet 文件。")

    print("    - 正在合并所有已加载的数据...")
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.dropna(subset=['id', 'title', 'text'], inplace=True)
    combined_df = combined_df.reset_index(drop=True)
    print(f"    - 合并后共有 {len(combined_df)} 篇文章。")

    if num_samples > 0 and num_samples < len(combined_df):
        print(f"    - 正在从中随机采样 {num_samples} 篇文章进行处理...")
        combined_df = combined_df.sample(n=num_samples, random_state=42).copy()

    print(f"    - 数据加载完成，最终处理的文章数量为: {len(combined_df)}")
    return combined_df

def worker_init_per_gpu(num_entities_to_extract: int):
    """
    多进程工作单元的初始化函数，为每个进程加载Spacy模型并分配GPU。
    """
    global nlp, NUM_ENTITIES_TO_EXTRACT
    NUM_ENTITIES_TO_EXTRACT = num_entities_to_extract # 将主进程的参数传递给全局变量
    import spacy
    worker_id = mp.current_process()._identity[0] - 1
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    num_gpus = len(visible_devices)
    gpu_to_use = worker_id % num_gpus
    # 1. 加载 Spacy
    try:
        if device.startswith('cuda'): spacy.prefer_gpu(gpu_index)
        # 原来的代码：
        # nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "lemmatizer"])
        
        # === 修改后的代码 ===
        nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "lemmatizer"])
        # 手动添加 sentencizer，专门用于断句，解决 ent.sent 报错问题
        nlp.add_pipe("sentencizer") 
        # ===================
        
    except Exception as e:
        print(f"    - [PID {os.getpid()}] Spacy GPU 失败: {e}，切换 CPU。")
        nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "lemmatizer"])
        # 同样记得在异常处理的 CPU 模式下也加上
        nlp.add_pipe("sentencizer")
def extract_specific_entities(text: str) -> Set[str]:
    """
    使用Spacy从文本中提取符合条件的命名实体。
    """
    global nlp, NUM_ENTITIES_TO_EXTRACT # 引用全局变量
    if nlp is None or not isinstance(text, str) or not text: return set()
    doc = nlp(text[:1_000_000]) # 限制处理文本长度以防止内存溢出
    specific_entities = set()
    for ent in doc.ents:
        entity_text = ent.text.strip()
        if (ent.label_ in VALID_ENTITY_LABELS and
                len(entity_text) > 3 and
                entity_text[0].isupper() and
                entity_text.split()[0].lower() not in SPECIFIC_ENTITY_STOPWORDS):
            specific_entities.add(entity_text)
            
    if len(specific_entities) > NUM_ENTITIES_TO_EXTRACT:
        return set(random.sample(list(specific_entities), NUM_ENTITIES_TO_EXTRACT))
    return specific_entities

def preprocess_text_for_matching(text: str) -> str:
    """
    对文本（实体或标题）进行预处理以进行匹配。
    转为小写并移除开头的冠词。
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    for article in {'the ', 'a ', 'an '}:
        if text.startswith(article):
            return text[len(article):]
    return text

def find_edges_for_doc(args: Tuple) -> List[Tuple]:
    """
    为单篇文章查找所有潜在的“精准匹配”边。
    """
    source_id, source_text, processed_title_map = args
    potential_edges = []
    source_entities = extract_specific_entities(source_text)
    if not source_entities: return []
    for entity in source_entities:
        processed_entity = preprocess_text_for_matching(entity)
        if processed_entity in processed_title_map:
            target_id = processed_title_map[processed_entity]
            if source_id != target_id:
                edge_data = {'bridge': entity, 'edit_distance': 0}
                potential_edges.append((source_id, target_id, edge_data))
    return potential_edges

def build_entity_title_graph(df: pd.DataFrame, processes_per_gpu: int, max_in_degree: int, num_entities_to_extract: int) -> nx.DiGraph:
    """
    通过实体与标题的精准匹配，并行构建知识图谱。
    """
    print(f"[*] 2/5: 正在通过“精准匹配”构建知识图谱...")
    print("    - 正在准备用于精准匹配的标题字典...")
    processed_title_map = {
        preprocess_text_for_matching(row['title']): row['id']
        for _, row in df.iterrows()
    }
    
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    num_processes = num_gpus * processes_per_gpu
    print(f"    - 检测到 {num_gpus} 个GPU, 每个GPU启动 {processes_per_gpu} 个进程, 总共启动 {num_processes} 个工作进程。")

    tasks = [(row['id'], row['text'], processed_title_map) for _, row in df.iterrows()]
    
    all_potential_edges = []
    # 使用 functools.partial 传递固定参数
    from functools import partial
    worker_initializer = partial(worker_init_per_gpu, num_entities_to_extract)

    with mp.Pool(processes=num_processes, initializer=worker_initializer) as pool:
        with tqdm(total=len(tasks), desc="    - 阶段1/2: 并行查找所有精准连接") as pbar:
            for result_edges in pool.imap_unordered(find_edges_for_doc, tasks):
                if result_edges:
                    all_potential_edges.extend(result_edges)
                pbar.update(1)

    print(f"    - 发现 {len(all_potential_edges)} 条潜在连接。")
    print(f"    - 阶段 2/2: 正在根据入度限制（每个桥梁实体最多 {max_in_degree} 次）构建最终图谱...")
    G = nx.DiGraph()
    G.add_nodes_from(df['id'])
    random.shuffle(all_potential_edges)
    
    bridge_in_degree_counts = {}
    for source_id, target_id, edge_data in tqdm(all_potential_edges, desc="    - 过滤并添加边"):
        bridge_entity = edge_data['bridge']
        counter_key = (target_id, bridge_entity.lower())
        current_count = bridge_in_degree_counts.get(counter_key, 0)
        if current_count < max_in_degree:
            G.add_edge(source_id, target_id, **edge_data)
            bridge_in_degree_counts[counter_key] = current_count + 1
            
    print("    - 图构建完成。")
    print(f"      - 节点数: {G.number_of_nodes()}")
    print(f"      - 边数 (连接数): {G.number_of_edges()}")
    return G

def generate_fixed_hop_path(graph: nx.DiGraph, start_node: str, num_hops: int) -> List[str]:
    """从图中的一个起始节点开始，随机游走生成一条固定跳数的路径。"""
    path = [start_node]
    current_node = start_node
    for _ in range(num_hops):
        # 确保不走回头路
        neighbors = [n for n in list(graph.successors(current_node)) if n not in path]
        if not neighbors: return None # 无法继续前进
        next_node = random.choice(neighbors)
        path.append(next_node)
        current_node = next_node
    return path

def package_path_to_custom_format(path: List[str], graph: nx.DiGraph, id_to_doc_map: Dict) -> Dict:
    """将路径列表格式化为指定的JSON对象结构。"""
    output_item = {}
    for i, doc_id in enumerate(path):
        output_item[f'document{i+1}'] = id_to_doc_map[doc_id]
        if i < len(path) - 1:
            next_doc_id = path[i+1]
            edge_data = graph.get_edge_data(doc_id, next_doc_id)
            output_item[f'connection_d{i+1}_d{i+2}'] = edge_data if edge_data else {'bridge': 'UNKNOWN'}
    return output_item

def is_path_valid(path: List[str], graph: nx.DiGraph, similarity_threshold: float) -> bool:
    """检查路径内的桥梁实体是否过于相似。"""
    if len(path) < 3: return True
    bridges = [graph.get_edge_data(path[i], path[i+1])['bridge'] for i in range(len(path) - 1)]
    if len(bridges) < 2: return True
    for bridge1, bridge2 in combinations(bridges, 2):
        if fuzz.ratio(bridge1, bridge2) > similarity_threshold:
            return False
    return True

def create_reasoning_paths_dataset(graph: nx.DiGraph, id_to_doc_map: Dict, num_paths: int, num_hops: int, similarity_threshold: float) -> List[Dict]:
    """从知识图谱中生成唯一的、符合条件的推理路径数据集。"""
    print(f"[*] 3/5: 正在生成 {num_paths} 条唯一的 {num_hops}-跳 推理路径...")
    if graph.number_of_edges() == 0:
        print("    - 警告: 图中没有边，无法生成路径。")
        return []
        
    dataset = []
    # 使用 (frozenset(文章ID), frozenset(小写桥梁)) 作为唯一性指纹
    seen_paths_fingerprints = set() 
    
    nodes_with_out_edges = [node for node, out_degree in graph.out_degree() if out_degree > 0]
    if not nodes_with_out_edges:
        print("    - 警告: 图中没有节点存在出边，无法生成路径。")
        return []
        
    with tqdm(total=num_paths, desc="    - 生成路径") as pbar:
        attempts = 0
        max_attempts = num_paths * 1500 # 增加尝试次数以应对更严格的去重
        
        while len(dataset) < num_paths and attempts < max_attempts:
            attempts += 1
            start_node = random.choice(nodes_with_out_edges)
            path = generate_fixed_hop_path(graph, start_node, num_hops)
            
            if not path: continue

            try:
                bridges = [graph.get_edge_data(path[i], path[i+1])['bridge'].lower() for i in range(len(path) - 1)]
            except (TypeError, KeyError):
                continue # 路径数据不完整，跳过

            path_fingerprint = (frozenset(path), frozenset(bridges))
            if path_fingerprint in seen_paths_fingerprints:
                continue
            
            if not is_path_valid(path, graph, similarity_threshold):
                continue
            
            seen_paths_fingerprints.add(path_fingerprint)
            formatted_path = package_path_to_custom_format(path, graph, id_to_doc_map)
            dataset.append(formatted_path)
            pbar.update(1)
            
    if len(dataset) < num_paths:
        print(f"\n    - 警告：尝试了 {attempts}/{max_attempts} 次，但只成功生成了 {len(dataset)}/{num_paths} 条路径。")
        
    return dataset

def main(args):
    """主执行函数"""
    # 1. 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.makedirs(args.output_path, exist_ok=True)
    
    # 2. 加载数据
    wiki_df = load_and_prepare_data(args.wiki_data_path, args.start_file_index, args.num_files, args.num_samples)
    id_to_doc_map = {row['id']: {'id': row['id'], 'title': row['title'], 'text': row['text']} for _, row in wiki_df.iterrows()}

    # 3. 构建或加载图
    graph_file = os.path.join(args.output_path, "knowledge_graph.gml")
    if os.path.exists(graph_file) and not args.force_rebuild_graph:
        print(f"[*] 2/5: 从文件 {graph_file} 加载已有的知识图谱...")
        knowledge_graph = nx.read_gml(graph_file)
    else:
        knowledge_graph = build_entity_title_graph(
            wiki_df,
            args.processes_per_gpu,
            args.max_in_degree_per_bridge,
            args.num_entities_to_extract
        )
        print(f"    - 正在将图保存到 {graph_file} 以便下次使用...")
        nx.write_gml(knowledge_graph, graph_file)

    # 4. 生成推理路径
    reasoning_paths_dataset = create_reasoning_paths_dataset(
        knowledge_graph, 
        id_to_doc_map, 
        args.num_paths_to_generate, 
        args.num_hops,
        args.bridge_similarity_threshold
    )

    # 5. 保存结果
    if reasoning_paths_dataset:
        # 文件名包含文章数和跳数，例如 3R_2hops.jsonl
        dataset_file = os.path.join(args.output_path, f"unique_{args.num_hops+1}R_{args.num_hops}hops.jsonl")
        print(f"[*] 4/5: 正在将 {len(reasoning_paths_dataset)} 条路径保存到 {dataset_file}...")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for item in reasoning_paths_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"[*] 5/5: 任务完成！")
    else:
        print("[*] 4/5: 没有生成任何路径，任务结束。")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从维基百科数据构建知识图谱并生成推理路径数据集。")
    
    # --- 路径和文件相关 ---
    parser.add_argument("--wiki_data_path", type=str, required=True, help="存放维基百科Parquet文件的目录路径。")
    parser.add_argument("--output_path", type=str, required=True, help="存放输出结果（图文件、数据集）的目录路径。")
    parser.add_argument("--start_file_index", type=int, default=0, help="从文件列表的哪个索引开始读取。")
    parser.add_argument("--num_files", type=int, default=1, help="总共读取多少个文件 (-1 表示读取到末尾)。")
    
    # --- 数据处理和采样 ---
    parser.add_argument("--num_samples", type=int, default=100000, help="从加载的文件中随机采样多少篇文章进行处理 (-1 表示全部使用)。")
    parser.add_argument("--num_entities_to_extract", type=int, default=50, help="从每篇文章中最多提取的实体数量。")
    
    # --- 图构建相关 ---
    parser.add_argument("--force_rebuild_graph", action='store_true', help="如果设置，即使存在已缓存的图文件，也强制重新构建图。")
    parser.add_argument("--max_in_degree_per_bridge", type=int, default=30, help="在图中，每个'桥梁实体'指向同一目标文章的最大入度限制。")

    # --- 数据集生成相关 ---
    parser.add_argument("--num_paths_to_generate", type=int, default=100000, help="要生成的目标唯一推理路径数量。")
    parser.add_argument("--num_hops", type=int, default=2, help="每条推理路径的跳数 (例如, 2跳路径包含3篇文章)。")
    parser.add_argument("--bridge_similarity_threshold", type=float, default=60.0, help="路径内桥梁实体的相似度阈值，高于此阈值的路径将被丢弃。")

    # --- 计算资源相关 ---
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="指定使用的GPU设备ID，用逗号分隔 (例如 '0,1')。")
    parser.add_argument("--processes_per_gpu", type=int, default=10, help="每个GPU上启动的工作进程数量。")

    return parser.parse_args()

if __name__ == "__main__":
    # 在Windows或macOS上，'spawn'是更安全的多进程启动方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("多进程启动方法已设置。")
    
    arguments = parse_args()
    main(arguments)
