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
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "text_output")
       
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
        
    def execute(self, requests):
        
        responses = []
        for request in requests:
            
            output_ids_tensor = pb_utils.get_input_tensor_by_name(request, "output_ids")
            output_ids = output_ids_tensor.as_numpy()
            

            outputs = self._postprocessing(output_ids)
            outputs = np.array(outputs).astype(self.output0_dtype)
            # prepare response object
            text_output_tensor  = pb_utils.Tensor("text_output", outputs)
            responses.append(pb_utils.InferenceResponse([text_output_tensor]))

        return responses

    def _postprocessing(self, tokens_batch):
        outputs = []
        for beam_tokens in tokens_batch:
            for tokens in beam_tokens:
                output = self.tokenizer.decode(tokens, skip_special_tokens=True)
                outputs.append(output.encode('utf8'))
        return outputs
