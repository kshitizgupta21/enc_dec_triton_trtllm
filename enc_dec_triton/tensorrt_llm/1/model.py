import torch
from transformers import (AutoConfig, AutoTokenizer)
import os
from run import TRTLLMEncDecModel
import json
import logging
import triton_python_backend_utils as pb_utils
from torch import from_numpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_input_tensor_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> PyTorch tensor
        return from_numpy(tensor.as_numpy())
    else:
        return tensor


def get_input_scalar_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> first scalar
        tensor = tensor.as_numpy()
        return tensor.reshape((tensor.size, ))[0]
    else:
        return tensor
        
class TritonPythonModel:
    # Every Python model must have "TritonPythonModel" as the class name!
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
        """
        
        # Read Max Batch Size from Triton Model config
        model_config = json.loads(args["model_config"])

        hf_model_dir = model_config['parameters']['hf_model_dir']['string_value']

        hf_model_config = AutoConfig.from_pretrained(hf_model_dir)
        self.decoder_start_token_id = hf_model_config.decoder_start_token_id
        self.bos_token_id = hf_model_config.bos_token_id
        self.eos_token_id = hf_model_config.eos_token_id
        self.pad_token_id = hf_model_config.pad_token_id
        # initialize TRT-LLM model
        engine_dir = model_config['parameters']['engine_dir']['string_value']
        engine_name = model_config['parameters']['engine_name']['string_value']
        self.tllm_model = TRTLLMEncDecModel.from_engine(engine_name, engine_dir)
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "output_ids")['data_type']
        )
        logger.info("TensorRT-LLM Enc-Dec Model in Python Backend initialized")


    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
            
        responses = []
        for request in requests:
            # extract input id and attention mask tensors from inpuy request
            input_ids = get_input_tensor_by_name(request, "input_ids").cuda()
            attention_mask = get_input_tensor_by_name(request, "attention_mask").cuda()
            max_new_tokens = get_input_scalar_by_name(request, "max_new_tokens")
            num_beams = 1
            beam_width = get_input_scalar_by_name(request, "beam_width")
            if beam_width is not None:
                num_beams = beam_width
            decoder_input_ids = torch.IntTensor([[self.decoder_start_token_id]]).cuda()
            batch_size = input_ids.shape[0]
            decoder_input_ids = decoder_input_ids.repeat((batch_size, 1))
            output_ids = self.tllm_model.generate(
                                encoder_input_ids=input_ids,
                                decoder_input_ids=decoder_input_ids,
                                max_new_tokens=max_new_tokens,
                                num_beams=num_beams,
                                bos_token_id=self.bos_token_id,
                                pad_token_id=self.pad_token_id,
                                eos_token_id=self.eos_token_id,
                                debug_mode=False,
                                return_dict=False,  # when set return_dict=True, get outputs by key
                                attention_mask=attention_mask)    
            output_ids = output_ids.cpu().numpy().astype(self.output_dtype)            
            output_ids_tensor = pb_utils.Tensor("output_ids", output_ids)
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_ids_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
