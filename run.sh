python GIANT_main.py \
--data_type concept \
--train_file "data\\concept\\concepts.json" \
--emb_tags \
--task_output_dims 2 \
--tasks "phrase" \
--edge_types_list "seq" "dep" "contain" "synonym" \
--d_model 32 \
--layers 3 \
--num_bases 5 \
--epochs 10 \
--mode train \
--debug

# sync test 2