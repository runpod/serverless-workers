## Dolly 2.0 Tuner

https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

The tunner is based on the [Dolly Repo](https://github.com/databrickslabs/dolly)

Model: https://huggingface.co/databricks/dolly-v2-12b

## Prepare Dataset

Datasets should be formatted to follow the [AlpacaDataset](https://github.com/gururise/AlpacaDataCleaned) format. The dataset is a collection of instruction, input, output statements formated in a JSON file with the following structure:

```JSON
[
    {
        "instruction": "instruction statement",
        "input": "input statement",
        "output": "output statement"
    }
]
```

Or the format instruction, context, response, category

```JSON
[
    {
        "instruction": "instruction statement",
        "context": "input statement",
        "response": "output statement",
        "category": "category"
    }
]
```
