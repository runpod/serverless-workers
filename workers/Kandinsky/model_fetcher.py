from kandinsky2 import get_kandinsky2
try:
    get_kandinsky2('cuda', task_type='text2img', model_version='2.1',
                   use_flash_attention=False, cache_dir="/kandinsky2",)
except RuntimeError as err:
    print(err)
