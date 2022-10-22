CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/text-classification/run_glue.py  \
 --model_name_or_path bert-base-uncased  --task_name sst-2 --data_dir data/sst-2/ \
   --do_train --do_eval   --max_seq_length 128   --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 5 \
    --output_dir outputs/sst_lwp/ --weight_poison 1 --save_steps 5000