# Endpoint Testing

Endpoint testing can help ensure that endpoints behave as expected. Since you can have virtually unlimited number of input paramaters for your endpoint, it becomes even more benifical to setup tests for your endpoint.

## Endpoint Test Data Structure

Create a directory under `endpoints` that has the same ID of the endpoint you will be testing. Within that directory you can add as many .json files that will contain your test inputs and expected outputs. The format of the .json file is as follows:

```json
{
    "TestInput":{
        "input": {
            "param1": "value1",
            "param2": "value2"
        }
    },
    "TestOutput":{
        "output": {
            "param1": "value1",
            "param2": "value2"
        }
    }
}
```
