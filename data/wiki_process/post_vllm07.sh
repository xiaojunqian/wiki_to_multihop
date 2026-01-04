#!/bin/bash

# ==========================================
# 核心配置参数
# ==========================================

# 端口配置
PORTS=(8110 8111 8112 8113 8114 8115 8116 8117)
NUM_WORKERS=${#PORTS[@]}

# 参数配置：$1 起始轮次, $2 结束轮次
START_ROUND=${1:-1}
END_ROUND=${2:-999999}

# 路径配置
NUM_HOPS=2
BASE_DATA_PATH="/mnt/cxzx/share/xiaojunqian/data"
BASE_OUTPUT_NAME="real_wiki_output_${NUM_HOPS}hops"

# 最终汇总文件路径
FINAL_QA_FILE="${BASE_DATA_PATH}/GLOBAL_1225_unique_${NUM_HOPS}hops_questions.jsonl"
FINAL_REF_FILE="${BASE_DATA_PATH}/GLOBAL_1225_unique_${NUM_HOPS}hops_references.jsonl"
FINAL_INV_FILE="${BASE_DATA_PATH}/GLOBAL_1225_unique_${NUM_HOPS}hops_invalid.jsonl"

# vLLM 启动脚本路径
SERVER_SCRIPT="/mnt/cxzx/share/xiaojunqian/scripts/multi_server_all.sh"
SERVER_SCRIPT_PID=""

WORKER_PIDS=()

# ==========================================
# 1. 生成辅助 Python 脚本 (端口检测 + 安全合并)
# ==========================================
generate_python_tools() {
# 1.1 端口检测工具
cat << 'EOF' > _temp_port_checker.py
import asyncio
import sys
from tqdm.asyncio import tqdm

async def check_port(ip, port, timeout_limit=600):
    waited = 0
    interval = 2
    while waited < timeout_limit:
        try:
            await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=1.0)
            return port
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(interval)
            waited += interval
    raise TimeoutError(f"Port {port}")

async def main():
    ports = [int(p) for p in sys.argv[1:]]
    if not ports: return
    print(f"🔍 [PortChecker] Monitoring ports: {ports}")
    tasks = [check_port('127.0.0.1', p) for p in ports]
    pbar = tqdm(total=len(ports), desc="🚀 Services Ready", unit="port", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    for f in asyncio.as_completed(tasks):
        try:
            port = await f
            pbar.update(1)
        except Exception as e:
            pbar.close()
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    pbar.close()
    print("\n✅ All ports are active.")

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: sys.exit(130)
EOF

# 1.2 安全合并工具 (已修复 read 报错问题)
cat << 'EOF' > _temp_safe_merger.py
import sys
import glob
import os

def safe_merge(target_file):
    # 找到所有的分片文件
    part_pattern = target_file + ".part*"
    # 按 part 后面的数字进行排序 (part0, part1, part10...)
    parts = glob.glob(part_pattern)
    if not parts:
        return
    
    # 按照 part 数字后缀排序，防止 part10 排在 part2 前面
    try:
        parts.sort(key=lambda x: int(x.split('.part')[-1]))
    except:
        parts.sort() # fallback

    print(f"📦 Merging {len(parts)} parts into {target_file}...")

    # 【修复】：使用 'ab+' 模式 (追加 + 读写)，允许在追加模式下读取内容
    with open(target_file, 'ab+') as outfile:
        # 1. 检查目标文件当前是否以换行符结尾，如果不是，补一个
        # 先移动指针到文件末尾
        outfile.seek(0, 2)
        
        # 如果文件不为空
        if outfile.tell() > 0:
            # 回退一个字节读取
            outfile.seek(-1, 2)
            last_char = outfile.read(1)
            # 如果最后一个字符不是换行符，写入一个换行符
            if last_char != b'\n':
                outfile.write(b'\n')

        # 2. 逐个合并分片
        for p_file in parts:
            if os.path.getsize(p_file) == 0:
                os.remove(p_file)
                continue
                
            with open(p_file, 'rb') as infile:
                data = infile.read()
                # 确保指针在末尾准备写入
                outfile.seek(0, 2)
                outfile.write(data)
                
                # 确保每个分片写完后，都有一个换行符分隔下一个分片
                if not data.endswith(b'\n'):
                    outfile.write(b'\n')
            
            # 3. 合并完一个立即删除一个 (防止重复)
            os.remove(p_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merger.py <target_file>")
        sys.exit(1)
    
    target = sys.argv[1]
    safe_merge(target)
EOF
}

# ==========================================
# 2. 清理函数
# ==========================================
cleanup_environment() {
    echo ""
    echo "=========================================="
    echo "🛑 [Cleanup] 任务结束，正在清理..."
    echo "=========================================="
    pkill -P $$ > /dev/null 2>&1
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    rm -f _temp_port_checker.py _temp_safe_merger.py
    echo "[Cleanup] ✅ 完成。"
}
trap cleanup_environment EXIT INT TERM

# ==========================================
# 3. 启动服务
# ==========================================
start_server() {
    generate_python_tools
    echo "[Service] 正在检查服务状态..."
    ALL_Active=true
    for port in "${PORTS[@]}"; do
        if ! nc -z localhost $port >/dev/null 2>&1; then
            ALL_Active=false; break
        fi
    done

    if [ "$ALL_Active" = true ]; then
        echo "✅ [Service] 端口已开启，复用服务。"
    else
        echo "⚠️  [Service] 启动服务: $SERVER_SCRIPT ..."
        bash "$SERVER_SCRIPT" > /dev/null 2>&1 &
        SERVER_SCRIPT_PID=$!
    fi

    echo "[Service] 等待就绪..."
    python3 _temp_port_checker.py "${PORTS[@]}"
    if [ $? -ne 0 ]; then exit 1; fi
}

# ==========================================
# 4. Worker 处理函数
# ==========================================
run_worker() {
    local worker_id=$1
    local port=${PORTS[$worker_id]}
    
    local worker_qa_file="${FINAL_QA_FILE}.part${worker_id}"
    local worker_ref_file="${FINAL_REF_FILE}.part${worker_id}"
    local worker_inv_file="${FINAL_INV_FILE}.part${worker_id}"
    local api_url="http://localhost:${port}/v1/chat/completions"

    echo "[Worker-$worker_id] 启动 (Port: $port)"
    local current_round=$((START_ROUND + worker_id))

    while [ $current_round -le $END_ROUND ]; do
        CURRENT_DIR="${BASE_DATA_PATH}/${BASE_OUTPUT_NAME}_${current_round}"
        
        if [ -d "${CURRENT_DIR}" ]; then
            INPUT_FILE=$(find "${CURRENT_DIR}" -name "*_hoped_qwened.jsonl" | head -n 1)
            if [ ! -z "$INPUT_FILE" ]; then
                echo "[Worker-$worker_id] Round ${current_round} 处理: $INPUT_FILE"
                
                # 下面的 Python 调用已经没有全角字符注释了
                python check_quality_client.py \
                    --input "$INPUT_FILE" \
                    --output-questions "$worker_qa_file" \
                    --output-references "$worker_ref_file" \
                    --output-invalid "$worker_inv_file" \
                    --history-file "$FINAL_QA_FILE" \
                    --api-url "$api_url" \
                    --concurrency 64 
            else
                # 只有目录存在但文件不存在时才输出，避免刷屏
                echo "[Worker-$worker_id] Round ${current_round} 目录存在但无文件，跳过。"
            fi
        fi
        
        current_round=$((current_round + NUM_WORKERS))
    done
}

# ==========================================
# 主流程
# ==========================================
start_server

echo "=========================================="
echo "🚀 并行任务开始: ${NUM_WORKERS} Workers"
echo "=========================================="

for i in "${!PORTS[@]}"; do
    run_worker "$i" &
    WORKER_PIDS+=($!)
done

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid"
done

echo "=========================================="
echo "📦 任务完成，正在安全合并分片..."
echo "=========================================="

# === 修复：使用 Python 工具进行安全合并 ===

# 1. 合并 Questions
python3 _temp_safe_merger.py "${FINAL_QA_FILE}"

# 2. 合并 References
python3 _temp_safe_merger.py "${FINAL_REF_FILE}"

# 3. 合并 Invalid
python3 _temp_safe_merger.py "${FINAL_INV_FILE}"

echo "✅ 所有流程结束。"
exit 0