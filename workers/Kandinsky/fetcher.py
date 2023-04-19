from kandinsky2 import get_kandinsky2
get_kandinsky2('cpu', task_type='text2img', model_version='2.1',
               use_flash_attention=False, cache_dir="/kandinsky2",)
