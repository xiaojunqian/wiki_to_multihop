python graph_spacy_R.py \
    --wiki_data_path "/mnt/cxzx/share/xiaojunqian/data/20231101.en" \
    --output_path "/mnt/cxzx/share/xiaojunqian/data/wiki_output_1" \
    --num_files 10 \
    --num_samples 130000 \
    --num_paths_to_generate 10000 \
    --num_hops 2 \
    --gpus "0,1" \
    --processes_per_gpu 8
sleep 60
python nR_deal.py \
    --input-file /mnt/cxzx/share/xiaojunqian/data/wiki_output_1/unique_3R_2hops.jsonl \
    --output-file /mnt/cxzx/share/xiaojunqian/data/wiki_output_1/unique_3R_2hops_hoped.jsonl \
    --max-tokens-per-snippet 1024
sleep 60
python generate_qa.py \
    --input_file /mnt/cxzx/share/xiaojunqian/data/wiki_output_1/unique_3R_2hops_hoped.jsonl \
    --output_path /mnt/cxzx/share/xiaojunqian/data/wiki_output_1/unique_3R_2hops_hoped_qwened.jsonl\
    --failed_output_path /mnt/cxzx/share/xiaojunqian/data/wiki_test_output/failed.jsonl \
    --prompt1_file /mnt/cxzx/share/xiaojunqian/data/wiki_process/prompt1_initializer_cn.md\
    --prompt2_file /mnt/cxzx/share/xiaojunqian/data/wiki_process/prompt2_replacer_cn.md \
    --model_name "qwen" \
    --num_workers 64 \
    --base_port 8110 \
    --num_endpoints 3