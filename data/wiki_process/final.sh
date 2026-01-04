#!/bin/bash

# ==============================================
# 配置区（根据实际需求修改！）
# ==============================================
ROOT_DIR="/mnt/cxzx/share/xiaojunqian/data"  # 数据根目录
DIR_PREFIX="real_wiki_output_"  # 目录前缀（real_或test_，按需切换）
# DIR_PREFIX="real_wiki_output_"  # 若目录是real_wiki_output_2hops_1，启用这行
HOPS_LIST="9 10 11 12 13 14 15 16 17"  # 要处理的hops列表（空格分隔！如1 2 或9 10 11）
INPUT_FILENAME="unique_3R_2hops_hoped_qwened.jsonl"  # 输入文件名（固定）
GLOBAL_QA_FILE="${ROOT_DIR}/global_qa.jsonl"  # 全局QA文件（JSONL）
GLOBAL_REF_FILE="${ROOT_DIR}/global_references.jsonl"  # 全局参考文件（JSONL）
INVALID_QA_FILE="${ROOT_DIR}/invalid_qa.jsonl"  # 无效数据文件（JSONL，存储answer在question中的原始数据）
PYTHON_SCRIPT="final.py"  # Python脚本路径（不在当前目录则写绝对路径）

# ==============================================
# 校验前置条件
# ==============================================
# 检查Python脚本是否存在
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "❌ 错误：未找到Python脚本 ${PYTHON_SCRIPT}"
    exit 1
fi

# 检查数据根目录是否存在
if [ ! -d "${ROOT_DIR}" ]; then
    echo "❌ 错误：数据根目录 ${ROOT_DIR} 不存在"
    exit 1
fi

# ==============================================
# 批量处理核心逻辑
# ==============================================
for hops in ${HOPS_LIST}; do
    echo "=================================================="
    echo "📂 开始处理：${DIR_PREFIX}2hops_${hops}"
    echo "=================================================="
    
    # 拼接完整输入路径
    INPUT_DIR="${ROOT_DIR}/${DIR_PREFIX}2hops_${hops}"  # 目录路径
    INPUT_FILE="${INPUT_DIR}/${INPUT_FILENAME}"        # 文件路径
    
    # 检查目录是否存在
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "⚠️  警告：目录 ${INPUT_DIR} 不存在，跳过"
        continue
    fi
    
    # 检查输入文件是否存在
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "⚠️  警告：文件 ${INPUT_FILE} 不存在，跳过该目录"
        continue
    fi
    
    # 调用Python脚本处理（新增--output-invalid参数）
    python3 "${PYTHON_SCRIPT}" \
        --input "${INPUT_FILE}" \
        --output-questions "${GLOBAL_QA_FILE}" \
        --output-references "${GLOBAL_REF_FILE}" \
        --output-invalid "${INVALID_QA_FILE}"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 处理成功！"
    else
        echo "❌ 处理失败！"
    fi
    
    echo -e "\n"
done

# ==============================================
# 处理完成总结
# ==============================================
echo "=================================================="
echo "🎉 所有目录处理完毕！"
echo "📋 总结："
echo "  - 处理的目录：${DIR_PREFIX}2hops_${HOPS_LIST}"
echo "  - 全局QA文件（JSONL）：${GLOBAL_QA_FILE}"
echo "  - 全局参考文件（JSONL）：${GLOBAL_REF_FILE}"
echo "  - 无效数据文件（JSONL）：${INVALID_QA_FILE}"
# 统计各文件记录数
if [ -f "${GLOBAL_QA_FILE}" ]; then
    QA_COUNT=$(grep -c . "${GLOBAL_QA_FILE}")
    echo "  - QA文件总记录数：${QA_COUNT}"
fi
if [ -f "${GLOBAL_REF_FILE}" ]; then
    REF_COUNT=$(grep -c . "${GLOBAL_REF_FILE}")
    echo "  - 参考文件总记录数：${REF_COUNT}"
fi
if [ -f "${INVALID_QA_FILE}" ]; then
    INVALID_COUNT=$(grep -c . "${INVALID_QA_FILE}")
    echo "  - 无效数据总记录数：${INVALID_COUNT}"
fi
echo "=================================================="