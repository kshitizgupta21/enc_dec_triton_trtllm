import json
import logging
import numpy as np 
import sys
import os
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """

    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                       legacy=False,
                                                       trust_remote_code=True)

        # Parse model output configs and convert Triton types to numpy types
        self.output_names = [
            "input_ids", "attention_mask"
        ]
        self.output_dtypes = {}
        for output_name in self.output_names:
            self.output_dtypes[output_name] = pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type'])


    def execute(self, requests):
                
        responses = []
        for request in requests:
            
            text_input = pb_utils.get_input_tensor_by_name(request,
                                                      'text_input').as_numpy()
            batch_size = text_input.shape[0]                        
            tok_batch = []
            for i in range(batch_size):                
                decoded_object = text_input[i, 0].decode()                                               
                tok_batch.append(decoded_object)
                            
            tok_sent = self.tokenizer(tok_batch, return_tensors="np", padding=True)       
            output_tensors = [] 
            for output_name in self.output_names:
                output = tok_sent.get(output_name).astype(self.output_dtypes[output_name])
                output_tensor = pb_utils.Tensor(output_name, output)
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(output_tensors = output_tensors)
            responses.append(inference_response)
        return responses
