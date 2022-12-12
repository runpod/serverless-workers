''' infer.py for runpod worker '''

import predict


class Predictor:
    ''' Predictor class for runpod worker '''

    def __init__(self):
        '''
        Initialize the predictor
        '''
        self.infer = None

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        self.infer = predict.Predictor()
        self.infer.setup()

    def predict(self, model_inputs):
        '''
        Run inference on the model
        '''
        return self.infer.predict(
            prompt=model_inputs["prompt"],
            width=model_inputs.get('width', 512),
            height=model_inputs.get('height', 512),
            init_image=model_inputs.get('init_image', None),
            mask=model_inputs.get('mask', None),
            prompt_strength=model_inputs.get('prompt_strength', 0.8),
            num_outputs=model_inputs.get('num_outputs', 1),
            num_inference_steps=model_inputs.get('num_inference_steps', 50),
            guidance_scale=model_inputs.get('guidance_scale', 7.5),
            scheduler=model_inputs.get('scheduler', "K-LMS"),
            seed=model_inputs.get('seed', None)
        )
