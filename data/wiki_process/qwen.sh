python generate_qa.py \
    --input_file /mnt/cxzx/share/xiaojunqian/data/new_wiki_output_2hops_16/unique_3R_2hops_hoped.jsonl \
    --output_path /mnt/cxzx/share/xiaojunqian/data/new_wiki_output_2hops_16/unique_3R_2hops_hoped_qwened.jsonl\
    --failed_output_path /mnt/cxzx/share/xiaojunqian/data/wiki_test_output/failed.jsonl \
    --prompt1_file /mnt/cxzx/share/xiaojunqian/data/wiki_process/prompt1_initializer_cn.md\
    --prompt2_file /mnt/cxzx/share/xiaojunqian/data/wiki_process/prompt2_replacer_cn.md \
    --model_name "qwen" \
    --num_workers 64 \
    --base_port 8110 \
    --num_endpoints 3