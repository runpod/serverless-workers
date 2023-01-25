'''
Used to fully test the expected behavior of the stable diffusion model endpoints.
'''

import os
import json
import time
import requests
from argparse import ArgumentParser


# ------------------------------ Parse Arguments ----------------------------- #
parser = ArgumentParser()
parser.add_argument('--key', type=str, help='RunPod API key')
parser.add_argument('--model', type=str, help='SD API Model name')
parser.add_argument('--dev', type=bool, help='Use dev API', default=False)

parser.add_argument('--input', type=str, help='Optional JSON input to test',
                    default=None)

args = parser.parse_args()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"{args.key}"
}

if args.dev:
    POST_URL = f'https://dev-api.runpod.ai/v1/{args.model}/run'
else:
    POST_URL = f'https://api.runpod.ai/v1/{args.model}/run'

if args.dev:
    GET_URL = f'https://dev-api.runpod.ai/v1/{args.model}/status'
else:
    GET_URL = f'https://api.runpod.ai/v1/{args.model}/status'

print()
print("Testing SD Model | " + args.model)
print("POST URL    | " + POST_URL)
print("GET URL     | " + GET_URL)


def run_test(job_in):
    '''
    Performs a single test of the SD model endpoint.
    '''
    # ------------------------------- Post Request ------------------------------- #
    request_response = requests.post(POST_URL, json={"input": job_in}, headers=headers, timeout=10)

    if request_response.status_code == 200:
        response_json = request_response.json()
        print("Job ID      | " + response_json["id"])
    else:
        print(request_response.status_code)
        print(request_response.text)

    # -------------------------------- Get Request ------------------------------- #
    status_check = requests.get(f"{GET_URL}/{response_json['id']}", headers=headers, timeout=10)

    if status_check.status_code == 200:
        status_json = status_check.json()
    else:
        print(status_check.status_code)
        print(status_check.text)

    if status_json["status"] == "IN_QUEUE":
        time_start = time.time()
        while status_json["status"] == "IN_QUEUE":
            time.sleep(1)
            status = requests.get(f"{GET_URL}/{response_json['id']}", headers=headers, timeout=10)
            status_json = status.json()
            print(
                f"Job Queued  | Time Elapsed: {time.time() - time_start} seconds",
                end='\r', flush=True)
        print()

    if status_json["status"] == "IN_PROGRESS":
        time_start = time.time()
        while status_json["status"] == "IN_PROGRESS":
            time.sleep(1)
            status = requests.get(f"{GET_URL}/{response_json['id']}", headers=headers, timeout=10)
            status_json = status.json()
            print(
                f"In Progress | Time Elapsed: {time.time() - time_start} seconds",
                end='\r', flush=True)
        print()

    if status_json["status"] == "COMPLETED":
        print("Test Result | Success")

    if status_json["status"] == "FAILED":
        print("Test Result | Fail")
        print("Job Error   | " + status_json["error"])


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if args.input:
    run_test(args.input)
else:
    sample_inputs = os.listdir("sample_inputs")
    for index, sample_input in enumerate(sample_inputs):
        print()
        print(f"Running Test {index + 1} of {len(sample_inputs)}")
        print("Test Input  | " + sample_input)
        with open(f'sample_inputs/{sample_input}', 'r', encoding="UTF-8") as sample_file:
            run_test(json.load(sample_file))
