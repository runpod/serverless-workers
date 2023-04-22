from transformers import AutoTokenizer, GPTJForCausalLM

AutoTokenizer.from_pretrained('databricks/dolly-v2-12b')
GPTJForCausalLM.from_pretrained('databricks/dolly-v2-12b')
