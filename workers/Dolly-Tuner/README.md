## Dolly 2.0 Tuner

The tunner is based on the [Dolly Repo](https://github.com/databrickslabs/dolly)

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
