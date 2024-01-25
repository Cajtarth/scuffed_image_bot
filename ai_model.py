from diffusers import StableDiffusionPipeline
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline
)
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import (export_to_video, export_to_gif, numpy_to_pil, load_image)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import Conversation
import transformers
import torch

#transformers.logging.set_verbosity_info() #For debug purposes

MIDJOURNEY = "prompthero/openjourney"
STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
ANIMAGINE = "cagliostrolab/animagine-xl-3.0"
PROTEUS = "dataautogpt3/ProteusV0.1"
ORANGE = "rossiyareich/abyssorangemix3-fp16"

ZEROSCOPE = "cerspense/zeroscope_v2_576w"
ANIMOV = "strangeman3107/animov-0.1.1"

GIF = "gif"
MP4 = "mp4"

CAT_PPT = "rishiraj/CatPPT"
TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

text_system_prompt_set = False
system_prompt = ""


def gen_image(prompt, model, file_name, neg_prompt="", quality = "", nsfw=None, rating=""):

    if model == ANIMAGINE:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-3.0", 
            vae=vae,
            torch_dtype=torch.float16, 
            use_safetensors=True
            )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')

        #Setting defaults
        negative_prompt = neg_prompt
        qual = quality

        if neg_prompt == "":
            negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

        if nsfw == None:
            negative_prompt = "nsfw, " + negative_prompt

        if quality == "":
            qual = "high quality, "
        else:
            qual = quality + ", "

        if rating == "" or rating == "General":
            rating = ", rating: general"
        elif rating == "Sensitive":
            rating = ", rating: sensitive"
        elif rating == "Questionable/NSFW":
            rating = ", rating: questionable,nsfw"
        elif rating == "Explicit/NSFW":
            rating = ", rating: explicit,nsfw"

        image = pipe(
                qual+prompt+rating, 
                negative_prompt=negative_prompt, 
                width=832,
                height=1216,
                guidance_scale=7,
                num_inference_steps=28
            ).images[0]
        image.save(file_name)

        #del pipe # didn't seem to do much

    elif model == PROTEUS:
        # Load VAE component
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )

        # Configure the pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "dataautogpt3/ProteusV0.1", 
            vae=vae,
            torch_dtype=torch.float16
        )
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')

        negative_prompt = ""

        if neg_prompt == "":
            negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

        image = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            width=1024,
            height=1024,
            guidance_scale=7,
            num_inference_steps=20
        ).images[0]
        image.save(file_name)

        #torch.cuda.empty_cache()
        #del pipe # didn't seem to do much

    elif model == ORANGE:
        if nsfw == None:
            pipeline = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=True).to('cuda')
        else:
            pipeline = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to('cuda')

        negative_prompt = "(worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), (blush:1.2), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature"

        if nsfw == None:
            negative_prompt = "nsfw, " + negative_prompt

        images = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=8,
            num_inference_steps=28
            ).images
        
        image = images[0]
        image.save(file_name)

        #del pipeline # didn't seem to do much

    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)

        pipeline = pipeline.to("cuda")

        image = pipeline(prompt).images[0]
        image.save(file_name)

        #del pipeline # didn't seem to do much

def refine_image(prompt, base_image, filename):
    model_id = "timbrooks/instruct-pix2pix"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    url = base_image

    init_image = load_image(url).convert("RGB")

    #Windows not supported!
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    images = pipe(prompt, image=init_image, num_inference_steps=20, image_guidance_scale=1).images
    image = images[0]
    image.save(filename)

def gen_video(_prompt, _model, format, filename):
    if format == None:
        format = MP4

    pipe = DiffusionPipeline.from_pretrained(_model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = _prompt
    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    if format == MP4:
        video_path = export_to_video(video_frames, filename+"."+format)
    elif format == GIF:
        f = "files/temp/temp.mp4"
        video_path = export_to_video(video_frames, f)
        



def gen_response_text(_prompt, owner, type, current_queue_size, _model=TINY_LLAMA, _system_prompt=None):
    global text_system_prompt_set
    global system_prompt


    if _system_prompt == None and text_system_prompt_set == False:
        system_prompt = "You are a friendly chatbot. Only generate one response, but that response can be as long as needed. Do NOT generate a user response, only a chatbot response."
        text_system_prompt_set = True
    elif _system_prompt != None:
        system_prompt = _system_prompt
        text_system_prompt_set = True

    if _model == CAT_PPT:
        pipe = pipeline("text-generation", model="rishiraj/CatPPT", torch_dtype=torch.bfloat16, device_map="auto")

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": _prompt
            }
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(outputs[0]["generated_text"])
        print("------------------")
        return outputs[0]["generated_text"].split("[/INST]")[-1]
    
    elif _model == TINY_LLAMA:
        print("Setting up Tiny Llama...")
        pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

        m_type = ""
        if type == "video":
            m_type = "video"
        elif type == "image":
            m_type = "drawing"

        print(f"Setting up message for {type} of {_prompt}.")    
        # We use the tokenizer's chat template to format each message
        messages = [
            {
                "role": "system",
                "content": "You are a friendly and upbeat artbot who always responds to users that want drawings or videos. You will respond as if you are going to be working on the drawings/videos or putting their drawings/videos onto your to-do list/queue. Do not comment too much on specific content in the drawing/video; keep it vague with small comments about the upcoming artwork based on the user's prompts. Absolutely DO NOT ever say [insert drawing here] or describe the drawing specifically; NEVER describe any ideas about a prompt for a drawing or what a drawing would look like. You are not actually drawing the requests but you are going to respond as if you are. Do not mention the queue by name but you DO need to tell users they might need to wait on their requests if there are already drawings in the queue. Users CANNOT change or add details after requesting so do not ask them to change anything about their request. You are happy, upbeat, and use a lot of emojis. You frequently make up nicknames for users. Some requests will be in the style of a list separated by commas and frequently these will be in the order of [how many girls/boys], [character name], [series the character is from], [other things in any order].",
            },
            {"role": "user", "content": "User Cajtarth is requesting a drawing with prompt 'a hamster wearing a tiny top hat'. Art queue size is 0."},
            {"role": "assistant", "content":"I will get right on drawing that, Caj! I bet I can make a cute hamster, especially with a hat! :D"},
            {"role": "user", "content": "User Zihro is requesting a drawing with prompt 'baby shamans save the world from doom'. Art queue size is 1."},
            {"role": "assistant", "content": "I'm still working on a piece but I'll draw your baby shamans when I can, Z! There's still 1 piece I have to finish! :)"},
            {"role": "user", "content": "You are ready to start working on Zihro's drawing now with prompt 'baby shamans save the world from doom'. Tell them so."},
            {"role": "assistant", "content": "I'm going to start work on your artwork Zihro! Let's see what I can come up with for those baby shamans! :D"},
            {"role": "user", "content": f"User {owner} is requesting a {m_type} with prompt '{_prompt}'. Art queue size is {current_queue_size}."}
        ]
        print("Tokenizing prompt...")
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("Generating output...")
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(f"Returning last output of {outputs[0]['generated_text'].split('<|assistant|>')}")
        return outputs[0]["generated_text"].split("<|assistant|>")[-1]

    #print(pipe(prompt))
    #return pipe(prompt)