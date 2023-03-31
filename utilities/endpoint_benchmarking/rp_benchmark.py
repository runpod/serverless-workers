'''
Endpoint Utilities | Benchmarking

Us to get a rough idea of how long it takes to run a particular endpoint.
'''

import time
import json
import requests
import argparse
from multiprocessing.pool import ThreadPool


def run_test(job_in):
    '''
    Performs a single test of the SD model endpoint.
    '''
    # convert job_in to json if it is a string
    job_in = job_in if isinstance(job_in, dict) else json.loads(job_in)

    print(f"Job Input   | {job_in}")
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

    return status_json


def benchmark_endpoint(job_in):
    '''
    Benchmark the endpoint with the given number of concurrent requests.
    '''
    # Baseline test
    baseline = run_test(job_in)

    baseline_queue_time = baseline["delayTime"] / 1000
    baseline_run_time = baseline["executionTime"] / 1000

    print(f"Baseline Queue Time: {baseline_queue_time} seconds")
    print(f"Baseline Run Time: {baseline_run_time} seconds")

    jobs_required_60s = int(60 / baseline_run_time)

    pool = ThreadPool(processes=jobs_required_60s)

    pool_constructor = [job_in for _ in range(jobs_required_60s)]

    time_start = time.time()
    pool_results = pool.map(run_test, pool_constructor)

    warm_boot_time = None
    previous_result_end_time = None
    for result in pool_results:
        if previous_result_end_time is not None and warm_boot_time is None:
            result_start_time = time_start + result["delayTime"] / 1000
            previous_result_end_time = result_start_time + result["executionTime"] / 1000

            if result_start_time < previous_result_end_time:
                warm_boot_time = result['delayTime'] / 1000

        print(f"Job Queue Time: {result['delayTime'] / 1000} seconds")
        print(f"Job Run Time: {result['executionTime'] / 1000} seconds")

        previous_result_end_time = time_start + \
            result["delayTime"] / 1000 + result["executionTime"] / 1000

    print(f"Total Time: {time.time() - time_start} seconds")
    print(f"Jobs Required: {jobs_required_60s}")
    print(f"Warm Boot Time: {warm_boot_time}s")


# ------------------------------ Parse Arguments ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--endpoint_id', type=str, help='The endpoint id')
parser.add_argument('--key', type=str, help='RunPod API key')
parser.add_argument('--dev', type=bool, help='Use dev API', default=False)

parser.add_argument('--input', type=str, help='Optional JSON input to test',
                    default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{args.key}"
    }

    if args.dev:
        POST_URL = f'https://dev-api.runpod.ai/v1/{args.endpoint_id}/run'
    else:
        POST_URL = f'https://api.runpod.ai/v1/{args.endpoint_id}/run'

    if args.dev:
        GET_URL = f'https://dev-api.runpod.ai/v1/{args.endpoint_id}/status'
    else:
        GET_URL = f'https://api.runpod.ai/v1/{args.endpoint_id}/status'

    benchmark_endpoint(args.input)
