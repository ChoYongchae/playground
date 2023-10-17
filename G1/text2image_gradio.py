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

def text2image(text):
    now = datetime.now()
    log_path = f"{now.year}_{now.month}_{now.date}.txt"
    with open(log_path, "w+", encoding="utf-8") as f:
        f.write(f"{text}\n")
    print(text)
    model = get_model()
    image = model(prompt=text).images[0]
    return image

demo = gr.Interface(
    fn=text2image,
    inputs=["생성할 텍스트를 입력해주세요. (약 5초 소요)"],
    outputs=["결과물"],
)
demo.launch(share=True)