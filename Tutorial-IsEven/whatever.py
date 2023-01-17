'''
This is the starting point for your serverless API.
It can be named anything you want, but needs to have the following:

- import runpod
- start function
'''

import runpod  # Required


def is_even(job_input):
    '''
    Example function that returns True if the input is even, False otherwise.

    "job_input" will contain the input that was passed to the API along with some other metadata.
    The structure will look like this:
    {
        "id": "some-id",
        "input": {"number": 2}
    }

    Whatever is returned from this function will be returned to the user as the output.
    '''

    job_input = job_input["input"]
    the_number = job_input["number"]

    if the_number % 2 == 0:
        return True
    else:
        return False


# This must be included for the serverless to work.
runpod.serverless.start({"handler": is_even})  # Required
