''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import download, validator, rp_cleanup


MODEL = predict.Predictor()
MODEL.setup()

INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': True
    },
    'model': {
        'type': str,
        'required': False
    },
    'transcription': {
        'type': str,
        'required': False
    },
    'translate': {
        'type': bool,
        'required': False
    },
    'language': {
        'type': str,
        'required': False
    },
    'temperature': {
        'type': float,
        'required': False
    },
    'best_of': {
        'type': int,
        'required': False
    },
    'beam_size': {
        'type': int,
        'required': False
    },
    'patience': {
        'type': float,
        'required': False
    },
    'length_penalty': {
        'type': float,
        'required': False
    },
    'suppress_tokens': {
        'type': str,
        'required': False
    },
    'initial_prompt': {
        'type': str,
        'required': False
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False
    },
    'temperature_increment_on_fallback': {
        'type': float,
        'required': False
    },
    'compression_ratio_threshold': {
        'type': float,
        'required': False
    },
    'logprob_threshold': {
        'type': float,
        'required': False
    },
    'no_speech_threshold': {
        'type': float,
        'required': False
    }
}


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Setting the float parameters
    job_input['temperature'] = float(job_input.get('temperature', 0))
    job_input['patience'] = float(job_input.get('patience', 0))
    job_input['length_penalty'] = float(job_input.get('length_penalty', 0))
    job_input['temperature_increment_on_fallback'] = float(
        job_input.get('temperature_increment_on_fallback', 0.2)
    )
    job_input['compression_ratio_threshold'] = float(
        job_input.get('compression_ratio_threshold', 2.4)
    )
    job_input['logprob_threshold'] = float(job_input.get('logprob_threshold', -1.0))
    job_input['no_speech_threshold'] = 0.6

    # Input validation
    input_errors = validator.validate(job_input, INPUT_VALIDATIONS)

    if input_errors:
        return {"error": input_errors}

    job_input['audio'] = download.download_input_objects([job_input['audio']])[0]

    whisper_results = MODEL.predict(
        audio=job_input["audio"],
        model=job_input.get("model", 'base'),
        transcription=job_input.get('transcription', 'plain_text'),
        translate=job_input.get('translate', False),
        language=job_input.get('language', None),
        temperature=job_input["temperature"],
        best_of=job_input.get("best_of", 5),
        beam_size=job_input.get("beam_size", 5),
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input.get('initial_prompt', None),
        condition_on_previous_text=job_input.get('condition_on_previous_text', True),
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
    )

    rp_cleanup.clean(['input_objects'])

    return {
        'segments': whisper_results.segments,
        'detected_language': whisper_results.detected_language,
        'transcription': whisper_results.transcription,
        'translation': whisper_results.translation
    }


runpod.serverless.start({"handler": run})
