python graph_spacy_R.py \
    --wiki_data_path "/mnt/cxzx/share/xiaojunqian/data/20231101.en" \
    --output_path "/mnt/cxzx/share/xiaojunqian/data/wiki_test_output_333" \
    --num_files 5 \
    --num_samples 300000 \
    --num_paths_to_generate 50000 \
    --num_hops 2 \
    --gpus "0,1" \
    --processes_per_gpu 15