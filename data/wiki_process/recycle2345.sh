#!/bin/bash

# ==========================================
# 核心配置参数
# ==========================================
# 在这里修改 Hops 数量，脚本会自动计算 R = HOPS + 1
NUM_HOPS=2
NUM_R=$((NUM_HOPS + 1))

# 基础路径
BASE_DATA_PATH="/mnt/cxzx/share/xiaojunqian/data"
PROMPT_DIR="${BASE_DATA_PATH}/wiki_process"  # Prompt 文件所在目录

# 日志目录 (必须提前创建)
SERVICE_LOG_DIR="/mnt/cxzx/share/xiaojunqian/scripts/logs"
mkdir -p "$SERVICE_LOG_DIR"

# 封装好的启动脚本路径 (针对 0,1,6,7 卡)
SERVER_STARTUP_SCRIPT="/mnt/cxzx/share/xiaojunqian/scripts/multi_server2345.sh"

# Prompt 文件名配置
PROMPT_FILE_1="prompt1_initializer_cn.md"
PROMPT_FILE_SAME="prompt_2_1_check_same.md"
PROMPT_FILE_GEN="prompt_2_2_generate.md"
PROMPT_FILE_UNIQUE="prompt_2_3_check_unique.md"

# 动态生成目录名
BASE_OUTPUT_NAME="real_wiki_output_${NUM_HOPS}hops"

# 动态生成文件前缀
FILE_PREFIX="unique_${NUM_R}R_${NUM_HOPS}hops"

# ==========================================
# 定义清理函数 (精准清理版：0,1,6,7)
# ==========================================
cleanup_environment() {
    echo "------------------------------------------"
    echo "[Cleanup] 开始执行精准清理..." 
    echo "          Target Ports: 8110, 8111"
    echo "          Target GPUs:  0, 1, 6, 7 (保护 2,3,4,5 不受影响)"

    # --- 1. 第一层：端口清理 (针对业务端口 8112, 8113) ---
    PORTS=(8110 8111)
    
    for port in "${PORTS[@]}"; do
        # lsof -t: 仅输出 PID
        PIDS=$(lsof -t -i :$port 2>/dev/null)
        
        if [ -n "$PIDS" ]; then
            echo "[Cleanup] 发现端口 $port 正在运行 (PID: $PIDS)，准备关闭..."
            
            # 尝试优雅关闭 (SIGTERM)
            echo "$PIDS" | xargs -r kill -15
            sleep 3
            
            # 检查是否仍存在，若存在则强制关闭 (SIGKILL)
            REMAINING_PIDS=$(lsof -t -i :$port 2>/dev/null)
            if [ -n "$REMAINING_PIDS" ]; then
                echo "[Cleanup] 端口 $port 进程仍存在，执行强制清理..."
                echo "$REMAINING_PIDS" | xargs -r kill -9
            fi
        else
            echo "[Cleanup] 端口 $port 此时空闲。"
        fi
    done

    # --- 2. 第二层：显卡物理清理 (仅针对 0,1,6,7) ---
    # [关键] 这里严格指定了显卡ID，不会误伤 2,3,4,5
    TARGET_GPUS=(2 3 4 5)
    
    if command -v fuser &> /dev/null; then
        for gpu_id in "${TARGET_GPUS[@]}"; do
            # fuser -v: 查看占用详情
            # /dev/nvidia${gpu_id}: 仅针对特定设备文件
            if fuser -v /dev/nvidia${gpu_id} >/dev/null 2>&1; then
                echo "[Cleanup] ⚠️ 发现 GPU ${gpu_id} 上有残留进程，正在强制清理..."
                # -k: kill, -9: SIGKILL
                fuser -k -9 -s /dev/nvidia${gpu_id}
            else
                echo "[Cleanup] GPU ${gpu_id} 干净。"
            fi
        done
    else
        echo "⚠️ [Warning] 未找到 'fuser' 命令。显卡物理清理跳过，仅依赖端口清理。"
    fi
    
    echo "[Cleanup] 清理流程结束，等待 5 秒系统同步..."
    sleep 5
    echo "------------------------------------------"
}

# 注册 Trap 信号 (脚本退出或被中断时自动清理)
trap cleanup_environment EXIT INT TERM

# ==========================================
# 主循环
# ==========================================
while true; do
    
    # --- 1. 寻找新的输出目录索引 ---
    i=1
    while [ -d "${BASE_DATA_PATH}/${BASE_OUTPUT_NAME}_${i}" ]; do
        i=$((i+1))
    done
    CURRENT_OUTPUT_DIR="${BASE_DATA_PATH}/${BASE_OUTPUT_NAME}_${i}"
    mkdir -p "${CURRENT_OUTPUT_DIR}"
    
    # 日志重定向 (同时输出到屏幕和 run.log)
    exec > >(tee -a "${CURRENT_OUTPUT_DIR}/run.log") 2>&1

    echo "=================================================="
    echo "开始新的一轮任务 (Round ${i})"
    echo "配置: Hops=${NUM_HOPS}, R=${NUM_R}"
    echo "启动脚本: ${SERVER_STARTUP_SCRIPT}"
    echo "GPU配置: Group1[2,3]@8110, Group2[4,5]@8111"
    echo "输出目录: ${CURRENT_OUTPUT_DIR}"
    echo "=================================================="

    # --- 0. 预清理环境 ---
    cleanup_environment

    # --- 2. 阶段 1: 生成路径 ---
    echo "[Step 1] 运行 graph_with_embedding.py..."
    # 确保此处调用显卡参数与 TARGET_GPUS 一致
    python -u graph_with_embedding.py \
        --wiki_data_path "${BASE_DATA_PATH}/20231101.en" \
        --output_path "${CURRENT_OUTPUT_DIR}" \
        --num_files 41 \
        --num_samples 1000000 \
        --num_paths_to_generate 100000 \
        --num_hops ${NUM_HOPS} \
        --gpus "2,3,4,5" \
        --processes_per_gpu 8
    
    # 检查 Step 1 结果
    EXPECTED_FILE_STEP1="${CURRENT_OUTPUT_DIR}/${FILE_PREFIX}.jsonl"
    
    if [ $? -ne 0 ] || [ ! -f "${EXPECTED_FILE_STEP1}" ]; then
        echo "❌ [Error] Step 1 失败或未生成文件: ${EXPECTED_FILE_STEP1}"
        echo "等待 60秒后重试..."
        sleep 60
        continue
    fi

    # --- 3. 阶段 2: 数据处理 ---
    echo "[Step 2] 运行 nR_deal.py..."
    python -u nR_deal.py \
        --input-file "${EXPECTED_FILE_STEP1}" \
        --output-file "${CURRENT_OUTPUT_DIR}/${FILE_PREFIX}_hoped.jsonl" \
        --max-tokens-per-snippet 1024
    
    if [ $? -ne 0 ]; then
        echo "❌ [Error] Step 2 失败。清理并跳过..."
        continue
    fi

    # --- 4. 启动服务 (使用封装脚本) ---
    echo "[Service] 调用封装脚本启动推理服务..."
    
    if [ -f "$SERVER_STARTUP_SCRIPT" ]; then
        # 后台运行启动脚本
        # 假设 multi_server0167.sh 内部处理了 nohup，或者会持续运行
        # 加上 & 确保主脚本能继续往下执行端口检测
        sh "$SERVER_STARTUP_SCRIPT" &
        
        echo "[Service] 启动命令已发送。日志位于: ${SERVICE_LOG_DIR}/vllm_server_*.log"
    else
        echo "❌ [Error] 找不到启动脚本: $SERVER_STARTUP_SCRIPT"
        exit 1
    fi
    
    # 等待端口就绪
    echo "[Service] 等待端口 8112 和 8113 就绪..."
    SERVICE_READY=0
    for attempt in {1..120}; do
        # 检查两个端口是否都在监听
        if nc -z localhost 8110 && nc -z localhost 8111; then
            echo "✅ [Service] 端口 8112 和 8113 均已连接！"
            SERVICE_READY=1
            break
        fi
        echo -n "."
        sleep 10
    done
    echo ""

    if [ $SERVICE_READY -eq 0 ]; then
        echo "❌ [Error] 服务启动超时 (1200秒内未就绪)。重试..."
        cleanup_environment
        continue
    fi
    
    # --- 5. 阶段 3: QA 生成 ---
    echo "[Step 3] 运行 qanew.PY..."
    python -u qanew.PY \
        --input_file "${CURRENT_OUTPUT_DIR}/${FILE_PREFIX}_hoped.jsonl" \
        --output_path "${CURRENT_OUTPUT_DIR}/${FILE_PREFIX}_hoped_qwened.jsonl" \
        --failed_output_path "${BASE_DATA_PATH}/wiki_test_output/failed_round_${i}.jsonl" \
        --prompt1_file "${PROMPT_DIR}/${PROMPT_FILE_1}" \
        --prompt_check_same_file "${PROMPT_DIR}/${PROMPT_FILE_SAME}" \
        --prompt_generate_file "${PROMPT_DIR}/${PROMPT_FILE_GEN}" \
        --prompt_check_unique_file "${PROMPT_DIR}/${PROMPT_FILE_UNIQUE}" \
        --model_name "qwen" \
        --num_workers 64 \
        --base_port 8110 \
        --num_endpoints 2

    if [ $? -eq 0 ]; then
        echo "✅ [Success] Round ${i} 全部完成！"
    else
        echo "❌ [Error] Step 3 失败。"
    fi

    echo "=================================================="
    
    # --- 6. 结束清理 (为下一轮做准备) ---
    cleanup_environment
    # 增加等待时间，确保端口完全释放
    sleep 10

done