export esnli_dir=???
export lirex_data_dir=???
export rationalizer_model_name=???          # use roberta-base or bert-base-uncased
export rationalizer_model_to_save=???       # *.pk
export rationalizer_best_model=???          # 
export rationalizer_output_dir=???
export generator_lm_output_dir=???          # directory to save the trained GPT2 model
export generator_gpt_model=???              # use gpt2-medium or gpt2
export selector_model_to_save=???           # *.pk
export best_selector_model=???              # 
export inference_model_to_save=???          # *.pk


########### prepare datasets
python create_datasets.py --data $esnli_dir/esnli_train_1.csv --output $lirex_data_dir/train_1.json
python create_datasets.py --data $esnli_dir/esnli_train_2.csv --output $lirex_data_dir/train_2.json
python create_datasets.py --data $esnli_dir/esnli_dev.csv --output $lirex_data_dir/dev.json
python create_datasets.py --data $esnli_dir/esnli_test.csv --output $lirex_data_dir/test.json

########### Rationalizer
python rationalizer/train.py --train_file $lirex_data_dir/train.json --dev_file $lirex_data_dir/dev.json --test_file $lirex_data_dir/test.json --model_name $rationalizer_model_name --model_to_save $rationalizer_model_to_save --lr 1e-5 -batch_size 32 --n_epoch 10

python rationalizer/predict.py --train_file $lirex_data_dir/train.json --dev_file $lirex_data_dir/dev.json --test_file $lirex_data_dir/test.json --model_name $rationalizer_model_name --model_to_load $rationalizer_best_model --output_dir $lirex_data_dir

########### Generator
python generator/prepare_data_for_finetune.py --data $lirex_data_dir/train.json --output $lirex_data_dir/train-finetune.txt
python generator/prepare_data_for_finetune.py --data $lirex_data_dir/dev.json --output $lirex_data_dir/dev-finetune.txt

python generator/GPT2_finetune_lm.py --output_dir $generator_lm_output_dir --model_type gpt2 --model_name_or_path $generator_gpt_model --do_train --train_data_file $lirex_data_dir/train-finetune.txt --do_eval --eval_data_file $lirex_data_dir/dev-finetune.txt --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --save_steps 1000 --save_total_limit 3

python generator/prepare_data_for_generation.py --data $lirex_data_dir/train-rationale.json --output $lirex_data_dir/train-prompts.txt
python generator/prepare_data_for_generation.py --data $lirex_data_dir/dev-rationale.json --output $lirex_data_dir/dev-prompts.txt
python generator/prepare_data_for_generation.py --data $lirex_data_dir/test-rationale.json --output $lirex_data_dir/test-prompts.txt

python generator/GPT2_generate.py --dataset $lirex_data_dir/train-prompts.txt --output $lirex_data_dir/train-gen.txt --model_type gpt2 --model_name_or_path $generator_lm_output_dir --length 30
python generator/GPT2_generate.py --dataset $lirex_data_dir/dev-prompts.txt --output $lirex_data_dir/dev-gen.txt --model_type gpt2 --model_name_or_path $generator_lm_output_dir --length 30
python generator/GPT2_generate.py --dataset $lirex_data_dir/test-prompts.txt --output $lirex_data_dir/test-gen.txt --model_type gpt2 --model_name_or_path $generator_lm_output_dir --length 30

########## Instance Selector and Inference 
python selector_and_inference/prepare_data_with_expl.py --data $lirex_data_dir/train-gen.txt --output $lirex_data_dir/train-expl.json
python selector_and_inference/prepare_data_with_expl.py --data $lirex_data_dir/dev-gen.txt --output $lirex_data_dir/dev-expl.json
python selector_and_inference/prepare_data_with_expl.py --data $lirex_data_dir/test-gen.txt --output $lirex_data_dir/test-expl.json

python selector_and_inference/train_selector.py --train_data $lirex_data_dir/train-expl.json --dev_data $lirex_data_dir/dev-expl.json --test_data $lirex_data_dir/test-expl.json --model_to_save $selector_model_to_save --lr 2e-5 --batch_size 64 --n_epoch 3

python selector_and_inference/train_inference.py --train_data $lirex_data_dir/train-expl.json --dev_data $lirex_data_dir/dev-expl.json --test_data $lirex_data_dir/test-expl.json --selector_model $best_selector_model --model_to_save $inference_model_to_save --lr 2e-5 --batch_size 64 --n_epoch 3


