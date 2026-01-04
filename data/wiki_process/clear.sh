#!/bin/bash

# ==============================================
# 配置区（与 batch_process.sh 保持一致！）
# ==============================================
ROOT_DIR="/mnt/cxzx/share/xiaojunqian/data"  # 数据根目录（必须和处理脚本一致）
GLOBAL_QA_FILE="${ROOT_DIR}/global_qa.jsonl"  # 全局QA文件
GLOBAL_REF_FILE="${ROOT_DIR}/global_references.jsonl"  # 全局参考文件
INVALID_QA_FILE="${ROOT_DIR}/invalid_qa.jsonl"  # 无效数据文件

# ==============================================
# 核心删除逻辑
# ==============================================
echo "=================================================="
echo "🗑️  全局生成文件删除工具"
echo "=================================================="
echo "即将删除以下3个文件（路径：${ROOT_DIR}）："
echo "1. ${GLOBAL_QA_FILE}"
echo "2. ${GLOBAL_REF_FILE}"
echo "3. ${INVALID_QA_FILE}"
echo -e "\n⚠️  警告：删除后数据无法恢复！"
read -p "是否继续？(y/N)：" confirm

# 确认判断（仅输入 y/Y 才执行删除）
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo -e "\n❌ 已取消删除，无文件变动。"
    exit 0
fi

echo -e "\n📥 开始执行删除..."
deleted_count=0
skipped_count=0

# 定义删除函数（兼容文件不存在的情况）
delete_file() {
    local file_path="$1"
    if [ -f "${file_path}" ]; then
        rm -f "${file_path}"  # -f 强制删除，避免文件不存在报错
        if [ $? -eq 0 ]; then
            echo "✅ 已删除：${file_path}"
            ((deleted_count++))
        else
            echo "❌ 删除失败：${file_path}（权限不足或文件被占用）"
            ((skipped_count++))
        fi
    else
        echo "ℹ️  跳过：${file_path}（文件不存在）"
        ((skipped_count++))
    fi
}

# 依次删除三个文件
delete_file "${GLOBAL_QA_FILE}"
delete_file "${GLOBAL_REF_FILE}"
delete_file "${INVALID_QA_FILE}"

# ==============================================
# 结果总结
# ==============================================
echo -e "\n=================================================="
echo "📊 删除完成！"
echo "  - 成功删除：${deleted_count} 个文件"
echo "  - 跳过/失败：${skipped_count} 个文件"
echo "=================================================="