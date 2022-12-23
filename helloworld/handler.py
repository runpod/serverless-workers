#!/usr/bin/env python

import runpod


def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print(event)

    # do the things

    return "Hello World"


runpod.serverless.start({"handler": handler})
