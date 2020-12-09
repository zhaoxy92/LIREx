export lirex_data_dir=???
export rationalizer_model_name=???          # use roberta-base or bert-base-uncased
export rationalizer_best_model=???          # 
export rationalizer_output_dir=???
export generator_lm_output_dir=???          # directory to save the trained GPT2 model
export generator_gpt_model=???              # use gpt2-medium or gpt2
export best_selector_model=???              # 
export best_inference_model=???             # 

########### Rationalizer

python rationalizer/predict_mnli.py --dev_matched $lirex_data_dir/multinli_dev_matched.jsonl --dev_mismatched $lirex_data_dir/multinli_dev_mismatched.jsonl --model_name $rationalizer_model_name --model_to_load $rationalizer_best_model --output_dir $lirex_data_dir

########### Generator

python generator/prepare_data_for_generation_mnli.py --data $lirex_data_dir/multinli-dev-matched-rationale.json --output $lirex_data_dir/multinli-dev-matched-prompts.txt
python generator/prepare_data_for_generation_mnli.py --data $lirex_data_dir/multinli-dev-mismatched-rationale.json --output $lirex_data_dir/multinli-dev-mismatched-prompts.txt

python generator/GPT2_generate.py --dataset $lirex_data_dir/multinli-dev-matched-prompts.txt --output $lirex_data_dir/multinli-dev-matched-gen.txt --model_type gpt2 --model_name_or_path $generator_lm_output_dir --length 30
python generator/GPT2_generate.py --dataset $lirex_data_dir/multinli-dev-mismatched-prompts.txt --output $lirex_data_dir/multinli-dev-mismatched-gen.txt --model_type gpt2 --model_name_or_path $generator_lm_output_dir --length 30


########## Instance Selector and Inference 
python selector_and_inference/prepare_data_with_expl.py --data $lirex_data_dir/multinli-dev-matched-gen.txt --output $lirex_data_dir/multinli-dev-matched-expl.json
python selector_and_inference/prepare_data_with_expl.py --data $lirex_data_dir/multinli-dev-mismatched-gen.txt --output $lirex_data_dir/multinli-dev-mismatched-expl.json

python selector_and_inference/predict.py --data $lirex_data_dir/multinli-dev-matched-expl.json --selector_model $best_selector_model --inference_model $best_inference_model --batch_size 64
python selector_and_inference/predict.py --data $lirex_data_dir/multinli-dev-mismatched-expl.json --selector_model $best_selector_model --inference_model $best_inference_model --batch_size 64

