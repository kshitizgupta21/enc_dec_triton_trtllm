###1. Get Triton TensorRT-LLM Backend 

Here we show approach of building the container

### 1. Get TRTLLM Backend repo
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
```

### 2. Build the Triton TRT-LLM backend container (this will also install `tensorrt_llm` automatically) but building the container can take a while.

```
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
```

### 3. Launch the docker container
```
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v $(pwd):/workspace -w /workspace  triton_trt_llm bash
```

### 4. For further steps see the README_nolora for regular enc-dec deployment and see README_lora for LoRa enc-dec Triton TRT-LLM deployment