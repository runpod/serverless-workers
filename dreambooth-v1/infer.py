''' Inference for the DreamBooth model. '''

import predictor


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
        self.infer = predictor.Predictor()
        self.infer.setup()

    def run(self, model_inputs):
        '''
        Run inference on the model
        '''
        return self.infer.predict(
            instance_prompt=model_inputs["instance_prompt"],
            class_prompt=model_inputs["class_prompt"],
            instance_data=model_inputs["instance_data"],
            class_data=model_inputs.get("class_data", None),
            num_class_images=model_inputs.get("num_class_images", 50),
            save_sample_prompt=model_inputs.get("save_sample_prompt", None),
            save_sample_negative_prompt=model_inputs.get("save_sample_negative_prompt", None),
            n_save_sample=model_inputs.get("n_save_sample", 1),
            save_guidance_scale=model_inputs.get("save_guidance_scale", 7.5),
            save_infer_steps=model_inputs.get("save_infer_steps", 50),
            pad_tokens=model_inputs.get("pad_tokens", False),
            with_prior_preservation=model_inputs.get("with_prior_preservation", True),
            prior_loss_weight=model_inputs.get("prior_loss_weight", 1.0),
            seed=model_inputs.get("seed", 512),
            resolution=model_inputs.get("resolution", 512),
            center_crop=model_inputs.get("center_crop", False),
            train_text_encoder=model_inputs.get("train_text_encoder", True),
            train_batch_size=model_inputs.get("train_batch_size", 1),
            sample_batch_size=model_inputs.get("sample_batch_size", 2),
            num_train_epochs=model_inputs.get("num_train_epochs", 1),
            max_train_steps=model_inputs.get("max_train_steps", 2000),
            gradient_accumulation_steps=model_inputs.get("gradient_accumulation_steps", 1),
            gradient_checkpointing=model_inputs.get("gradient_checkpointing", False),
            learning_rate=model_inputs.get("learning_rate", 1e-6),
            scale_lr=model_inputs.get("scale_lr", False),
            lr_scheduler=model_inputs.get("lr_scheduler", "constant"),
            lr_warmup_steps=model_inputs.get("lr_warmup_steps", 0),
            use_8bit_adam=model_inputs.get("use_8bit_adam", True),
            adam_beta1=model_inputs.get("adam_beta1", 0.9),
            adam_beta2=model_inputs.get("adam_beta2", 0.999),
            adam_weight_decay=model_inputs.get("adam_weight_decay", 1e-2),
            adam_epsilon=model_inputs.get("adam_epsilon", 1e-8),
            max_grad_norm=model_inputs.get("max_grad_norm", 1.0),
        )
