## Instructions for running enc-dec model in Triton TRT-LLM through Python backend

### 1. Optimize model with TRT-LLM
1. Place the BART/T5 model in `hf_models`. Currently, only `pytorch_model.bin` format is supported and `safetensors` format is not supported yet for enc-dec. Here we show T5 small model in this example.
```
git clone https://huggingface.co/t5-small hf_models/t5-small
```

3. Convert HF checkpoint to TRT checkpoint format
```
python tensorrtllm_backend/tensorrt_llm/examples/enc_dec/t5/convert.py -i hf_models/t5-small -o trt_checkpoints/t5/ --weight_data_type float16 --inference_tensor_para_size 1
```

3. Build TRT-LLM engine
```
python tensorrtllm_backend/tensorrt_llm/examples/enc_dec/build.py --model_type t5 --weight_dir trt_checkpoints/t5/tp1 -o trt_engines/t5/1-gpu/ --engine_name t5 --use_bert_attention_plugin --use_gpt_attention_plugin --use_gemm_plugin --dtype float16 --max_beam_width 1 --remove_input_padding
```

## 2. Deploy TRT-LLM engine through Triton Python backend
### 1. Create Triton Model Repository and copy example config and scripts
```
mkdir triton_model_repo
cp -r enc_dec_triton/preprocessing triton_model_repo/
cp -r enc_dec_triton/postprocessing triton_model_repo/
cp -r enc_dec_triton/ensemble triton_model_repo/
cp -r enc_dec_triton/tensorrt_llm triton_model_repo/

```

#### copy enc_dec python runtime script into model repository
```
cp tensorrtllm_backend/tensorrt_llm/examples/enc_dec/run.py triton_model_repo/tensorrt_llm/1/
```

### 2. Modify the model configuration
#### Prepare preprocessing config.pbtxt
The following table shows the fields that need to be modified before deployment:

*triton_model_repo/preprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 8 |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/workspace/hf_models/t5-small`|
| `preprocessing_instance_count` | Here setting to 1 |


```
export HF_MODEL=/workspace/hf_models/t5-small
python3 fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:8,tokenizer_dir:${HF_MODEL},preprocessing_instance_count:1"
```
#### Pepare tensorrt_llm model config.pbtxt

*triton_model_repo/tensorrt_llm/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 8 |
| `engine_dir` | The path to TRT-LLM engines, here setting to `/workspace/trt_engines/t5/1-gpu/float16/tp1/`|
| `engine_name` | The name of the engine, also specified during engine build time. Here setting to `t5` |
| `hf_model_dir` | Huggingface model directory |


```
export HF_MODEL=/workspace/hf_models/t5-small
python3 fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:8,engine_dir:/workspace/trt_engines/t5/1-gpu/float16/tp1/,engine_name:t5,hf_model_dir:${HF_MODEL}"
```

#### Prepare postprocessing config.pbtxt

*triton_model_repo/postprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 8 |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/workspace/hf_models/t5-small`|
| `postprocessing_instance_count` | Here setting to 1 |

```
export HF_MODEL=/workspace/hf_models/t5-small
python3 fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:8,tokenizer_dir:${HF_MODEL},postprocessing_instance_count:1"
```

#### Run the following command to prepare ensemble config.pbtxt
*triton_model_repo/ensemble/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 8 |

```
python3 fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:8"
```

### 3. Launch Triton server

Run this command to launch Triton server.

```
tritonserver --model-repository triton_model_repo  
```

### 4. Query the server with the Triton generate endpoint
[Query the server with the Triton generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend#query-the-server-with-the-triton-generate-endpoint)

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Translate English to German: This is good", "max_new_tokens": 50}'
```






















