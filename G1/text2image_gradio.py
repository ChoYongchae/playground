import gradio as gr
from diffusers import StableDiffusionPipeline
from datetime import datetime

# TODO: make class
log_path = ""
models = {}
def get_model(model_name="runwayml/stable-diffusion-v1-5"):
    import torch
    if model_name not in models:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16) 
        pipe.to("cuda")
        models[model_name] = pipe
        # TODO: management gpu memory
    return models[model_name]

def text2image(input_text):
    now = datetime.now()
    log_path = f"{now.year}_{now.month}_{now.date}.txt"
    with open(log_path, "w+", encoding="utf-8") as f:
        f.write(f"{input_text}\n")
    print(input_text)
    model = get_model()
    generated = model(prompt=input_text).images[0]
    return generated

# TODO: blocks 사용해보기
demo = gr.Interface(
    fn=text2image,
    inputs=["text"],
    outputs=["image"],
)
demo.launch(share=True)