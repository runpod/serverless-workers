# Utilities

A collection of utilities that can be used to support endpoint development and deployment.

## Endpoint Testing

The `endpoint_test.py` script can be used to test an endpoint. It can be used to test the endpoint locally or remotely.

## Endpoint Benchmarking

The `endpoint_benchmark.py` script can be used to benchmark an endpoint.

Inputs:
- Endpoint ID
- API Key
- Min workers
- Max workers

Returns:

- The average job time in seconds
- Min time job spent in queue
- Max time job spent in queue
- Estimated warm boot time
- Estimated cold boot time

For best results, set your min workers to 1 and max workers to 5. Ensure no additional jobs are running on the endpoint during the benchmark.
