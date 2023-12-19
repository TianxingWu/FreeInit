import os
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationFreeInitPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from diffusers.training_utils import set_seed

from animatediff.utils.freeinit_utils import get_freq_filter
from collections import namedtuple

pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
inference_config_path = "configs/inference/inference-v1.yaml"

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

examples = [
    # 0-RealisticVision
    [
        "realisticVisionV51_v20Novae.safetensors", 
        "mm_sd_v14.ckpt", 
        "A panda standing on a surfboard in the ocean under moonlight.",
        "worst quality, low quality, nsfw, logo",
        512, 512, "2005563494988190",
        "butterworth", 0.25, 0.25, 3,
        ["use_fp16"]
    ],
    # 1-ToonYou
    [
        "toonyou_beta3.safetensors", 
        "mm_sd_v14.ckpt", 
        "(best quality, masterpiece), 1girl, looking at viewer, blurry background, upper body, contemporary, dress",
        "(worst quality, low quality)",
        512, 512, "478028150728261",
        "butterworth", 0.25, 0.25, 3,
        ["use_fp16"]
    ],
    # 2-Lyriel
    [
        "lyriel_v16.safetensors", 
        "mm_sd_v14.ckpt", 
        "hypercars cyberpunk moving, muted colors, swirling color smokes, legend, cityscape, space",
        "3d, cartoon, anime, sketches, worst quality, low quality, nsfw, logo",
        512, 512, "1566149281915957",
        "butterworth", 0.25, 0.25, 3,
        ["use_fp16"]
    ],
    # 3-RCNZ
    [
        "rcnzCartoon3d_v10.safetensors", 
        "mm_sd_v14.ckpt", 
        "A cute raccoon playing guitar in a boat on the ocean",
        "worst quality, low quality, nsfw, logo",
        512, 512, "1566149281915957",
        "butterworth", 0.25, 0.25, 3,
        ["use_fp16"]
    ],
    # 4-MajicMix
    [
        "majicmixRealistic_v5Preview.safetensors", 
        "mm_sd_v14.ckpt", 
        "1girl, reading book",
        "(ng_deepnegative_v1_75t:1.2), (badhandv4:1), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, watermark, moles",
        512, 512, "2005563494988190",
        "butterworth", 0.25, 0.25, 3,
        ["use_fp16"]
    ],
    # # 5-RealisticVision
    # [
    #     "realisticVisionV51_v20Novae.safetensors", 
    #     "mm_sd_v14.ckpt", 
    #     "A panda standing on a surfboard in the ocean in sunset.",
    #     "worst quality, low quality, nsfw, logo",
    #     512, 512, "2005563494988190",
    #     "butterworth", 0.25, 0.25, 3,
    #     ["use_fp16"]
    # ]
]

# clean unrelated ckpts
# ckpts = [
#     "realisticVisionV40_v20Novae.safetensors",
#     "majicmixRealistic_v5Preview.safetensors",
#     "rcnzCartoon3d_v10.safetensors",
#     "lyriel_v16.safetensors",
#     "toonyou_beta3.safetensors"
# ]

# for path in glob(os.path.join("models", "DreamBooth_LoRA", "*.safetensors")):
#     for ckpt in ckpts:
#         if path.endswith(ckpt): break
#     else:
#         print(f"### Cleaning {path} ...")
#         os.system(f"rm -rf {path}")

# os.system(f"rm -rf {os.path.join('models', 'DreamBooth_LoRA', '*.safetensors')}")

# os.system(f"bash download_bashscripts/1-ToonYou.sh")
# os.system(f"bash download_bashscripts/2-Lyriel.sh")
# os.system(f"bash download_bashscripts/3-RcnzCartoon.sh")
# os.system(f"bash download_bashscripts/4-MajicMix.sh")
# os.system(f"bash download_bashscripts/5-RealisticVision.sh")

# clean Gradio cache
print(f"### Cleaning cached examples ...")
os.system(f"rm -rf gradio_cached_examples/")


class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir                = os.path.join(self.basedir, "samples")
        os.makedirs(self.savedir, exist_ok=True)

        self.base_model_list    = []
        self.motion_module_list = []
        self.filter_type_list = [
            "butterworth",
            "gaussian",
            "box",
            "ideal"
        ]
        
        self.selected_base_model    = None
        self.selected_motion_module = None
        self.selected_filter_type = None
        self.set_width = None
        self.set_height = None
        self.set_d_s = None
        self.set_d_t = None
        
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # config models
        self.inference_config      = OmegaConf.load(inference_config_path)

        self.tokenizer             = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder          = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda()
        self.vae                   = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda()
        self.unet                  = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()

        self.freq_filter = None

        self.update_base_model(self.base_model_list[-2])
        self.update_motion_module(self.motion_module_list[0])
        self.update_filter(512, 512, self.filter_type_list[0], 0.25, 0.25)
        
        
    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = sorted([os.path.basename(p) for p in motion_module_list])

    def refresh_personalized_model(self):
        base_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.base_model_list = sorted([os.path.basename(p) for p in base_model_list])


    def update_base_model(self, base_model_dropdown):
        self.selected_base_model = base_model_dropdown
        
        base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
        base_model_state_dict = {}
        with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys(): base_model_state_dict[key] = f.get_tensor(key)
                
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown
        
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        _, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
        return gr.Dropdown.update()
    
    # def update_filter(self, shape, method, n, d_s, d_t):
    def update_filter(self, width_slider, height_slider, filter_type_dropdown, d_s_slider, d_t_slider):
        self.set_width = width_slider
        self.set_height = height_slider
        self.selected_filter_type = filter_type_dropdown
        self.set_d_s = d_s_slider
        self.set_d_t = d_t_slider

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        shape = [1, 4, 16, self.set_width//vae_scale_factor, self.set_height//vae_scale_factor]
        self.freq_filter = get_freq_filter(
            shape, 
            device="cuda", 
            filter_type=self.selected_filter_type,
            n=4,
            d_s=self.set_d_s,
            d_t=self.set_d_t
        )

    def animate(
        self,
        base_model_dropdown,
        motion_module_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        width_slider,
        height_slider,
        seed_textbox,
        # freeinit params
        filter_type_dropdown,
        d_s_slider,
        d_t_slider,
        num_iters_slider,
        # speed up
        speed_up_options
    ):
        # set global seed
        set_seed(42)

        d_s = float(d_s_slider)
        d_t = float(d_t_slider)
        num_iters = int(num_iters_slider)


        if self.selected_base_model != base_model_dropdown: self.update_base_model(base_model_dropdown)
        if self.selected_motion_module != motion_module_dropdown: self.update_motion_module(motion_module_dropdown)
        
        self.set_width = width_slider
        self.set_height = height_slider
        self.selected_filter_type = filter_type_dropdown
        self.set_d_s = d_s
        self.set_d_t = d_t
        if self.set_width != width_slider or self.set_height != height_slider or self.selected_filter_type != filter_type_dropdown or self.set_d_s != d_s or self.set_d_t != d_t:
            self.update_filter(width_slider, height_slider, filter_type_dropdown, d_s, d_t)
        
        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationFreeInitPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
            ).to("cuda")
        
        # (freeinit) initialize frequency filter for noise reinitialization -------------
        pipeline.freq_filter = self.freq_filter
        # -------------------------------------------------------------------------------

        
        if int(seed_textbox) > 0: seed = int(seed_textbox)
        else: seed = random.randint(1, 1e16)
        torch.manual_seed(int(seed))
        
        assert seed == torch.initial_seed()
        print(f"### seed: {seed}")
        
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
               
        sample_output = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = 25,
            guidance_scale      = 7.5,
            width               = width_slider,
            height              = height_slider,
            video_length        = 16,
            num_iters           = num_iters,
            use_fast_sampling   = True if "use_coarse_to_fine_sampling" in speed_up_options else False,
            save_intermediate   = False,
            return_orig         = True,
            use_fp16            = True if "use_fp16" in speed_up_options else False
        )
        orig_sample = sample_output.orig_videos
        sample = sample_output.videos

        save_sample_path = os.path.join(self.savedir, f"sample.mp4")
        save_videos_grid(sample, save_sample_path)

        save_orig_sample_path = os.path.join(self.savedir, f"sample_orig.mp4")
        save_videos_grid(orig_sample, save_orig_sample_path)

        # save_compare_path = os.path.join(self.savedir, f"compare.mp4")
        # save_videos_grid(torch.concat([orig_sample, sample]), save_compare_path)
    
        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "base_model": base_model_dropdown,
            "motion_module": motion_module_dropdown,
            "filter_type": filter_type_dropdown,
            "d_s": d_s,
            "d_t": d_t,
            "num_iters": num_iters,
            "use_fp16": True if "use_fp16" in speed_up_options else False,
            "use_coarse_to_fine_sampling": True if "use_coarse_to_fine_sampling" in speed_up_options else False
        }

        # return gr.Video.update(value=save_compare_path), gr.Json.update(value=json_config)
        # return gr.Video.update(value=save_orig_sample_path), gr.Video.update(value=save_sample_path), gr.Video.update(value=save_compare_path), gr.Json.update(value=json_config)
        return gr.Video.update(value=save_orig_sample_path), gr.Video.update(value=save_sample_path), gr.Json.update(value=json_config)
        

controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        # gr.Markdown('# FreeInit')
        gr.Markdown(
            """
            <div align="center">
            <h1>FreeInit</h1>
            </div>
            """
        )
        gr.Markdown(
            """
            <p align="center">
                    <a title="Project Page" href="https://tianxingwu.github.io/pages/FreeInit/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493">
                    </a>
                    <a title="arXiv" href="https://arxiv.org/abs/2312.07537" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b">
                    </a>
                    <a title="GitHub" href="https://github.com/TianxingWu/FreeInit" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/TianxingWu/FreeInit?label=GitHub%20%E2%98%85&&logo=github" alt="badge-github-stars">
                    </a>
                    <a title="Video" href="https://youtu.be/lS5IYbAqriI" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red">
                    </a>
                    <a title="Visitor" href="https://hits.seeyoufarm.com" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fhuggingface.co%2Fspaces%2FTianxingWu%2FFreeInit&count_bg=%23678F74&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false">
                    </a>
            </p>
            """
            # <a title="Visitor" href="https://hits.seeyoufarm.com" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            #     <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fhuggingface.co%2Fspaces%2FTianxingWu%2FFreeInit&count_bg=%23678F74&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false">
            # </a>
        )
        gr.Markdown(
            """
            Official Gradio Demo for ***FreeInit: Bridging Initialization Gap in Video Diffusion Models***.
            FreeInit improves time consistency of diffusion-based video generation at inference time. In this demo, we apply FreeInit on [AnimateDiff v1](https://github.com/guoyww/AnimateDiff) as an example. Sampling time: ~ 80s.<br>
            """
        )

        with gr.Row():
            with gr.Column():
                # gr.Markdown(
                #     """
                #     ### Usage
                #     1. Select customized model and motion module in `Model Settings`.
                #     3. Set `FreeInit Settings`.
                #     3. Provide `Prompt` and `Negative Prompt` for your selected model. You can refer to each model's webpage on CivitAI to learn how to write prompts for them:
                #         - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                #         - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                #         - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                #         - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)
                #         - [`realisticVisionV20_v20.safetensors`](https://civitai.com/models/4201?modelVersionId=29460)
                #     4. Click `Generate`.
                #     """
                # )
                prompt_textbox          = gr.Textbox( label="Prompt",          lines=3, placeholder="Enter your prompt here")
                negative_prompt_textbox = gr.Textbox( label="Negative Prompt", lines=3, value="worst quality, low quality, nsfw, logo")

                gr.Markdown(
                    """
                    *Prompt Tips:*

                    For each personalized model in `Model Settings`, you can refer to their webpage on CivitAI to learn how to write good prompts for them:
                    - [`realisticVisionV51_v20Novae.safetensors`](https://civitai.com/models/4201?modelVersionId=130072)
                    - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                    - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                    - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                    - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)   
                    """
                )
                
                with gr.Accordion("Model Settings", open=False):
                    gr.Markdown(
                        """
                        Select personalized model and motion module for AnimateDiff.
                        """
                        )
                    base_model_dropdown     = gr.Dropdown( label="Base DreamBooth Model", choices=controller.base_model_list,    value=controller.base_model_list[-2],    interactive=True,
                                                          info="Select personalized text-to-image model from community")
                    motion_module_dropdown  = gr.Dropdown( label="Motion Module",  choices=controller.motion_module_list, value=controller.motion_module_list[0], interactive=True,
                                                          info="Select motion module. Recommend mm_sd_v14.ckpt for larger movements.")
                
                base_model_dropdown.change(fn=controller.update_base_model,       inputs=[base_model_dropdown],    outputs=[base_model_dropdown])
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])
                
                with gr.Accordion("FreeInit Params", open=False):
                    gr.Markdown(
                        """
                        Adjust to control the smoothness.
                        """
                    )
                    filter_type_dropdown    = gr.Dropdown( label="Filter Type",  choices=controller.filter_type_list, value=controller.filter_type_list[0], interactive=True, 
                                                          info="Default as Butterworth. To fix large inconsistencies, consider using Gaussian.")
                    d_s_slider             = gr.Slider( label="d_s",  value=0.25, minimum=0, maximum=1, step=0.125, 
                                                       info="Stop frequency for spatial dimensions (0.0-1.0)")
                    d_t_slider             = gr.Slider( label="d_t",  value=0.25, minimum=0, maximum=1, step=0.125, 
                                                       info="Stop frequency for temporal dimension (0.0-1.0)")
                    # num_iters_textbox       = gr.Textbox( label="FreeInit Iterations", value=3, info="Sould be integer >1, larger value leads to smoother results)")
                    num_iters_slider        = gr.Slider( label="FreeInit Iterations", value=3, minimum=2, maximum=5, step=1,
                                                        info="Larger value leads to smoother results & longer inference time.")

                with gr.Accordion("Advance", open=False):
                    with gr.Row():
                        width_slider  = gr.Slider(  label="Width",  value=512, minimum=256, maximum=1024, step=64 )
                        height_slider = gr.Slider(  label="Height", value=512, minimum=256, maximum=1024, step=64 )
                    with gr.Row():
                        seed_textbox = gr.Textbox( label="Seed",  value=1566149281915957)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e16)), inputs=[], outputs=[seed_textbox])
                    with gr.Row():
                        speed_up_options = gr.CheckboxGroup(
                            ["use_fp16", "use_coarse_to_fine_sampling"],
                            label="Speed-Up Options",
                            value=["use_fp16"]
                        )


                generate_button = gr.Button( value="Generate", variant='primary' )


            # with gr.Column():
            #     result_video = gr.Video( label="Generated Animation", interactive=False )
            #     json_config  = gr.Json( label="Config", value=None )
            with gr.Column():
                with gr.Row():
                    orig_video = gr.Video( label="AnimateDiff", interactive=False )
                    freeinit_video = gr.Video( label="AnimateDiff + FreeInit", interactive=False )
                # with gr.Row():
                #     compare_video = gr.Video( label="Compare", interactive=False )
                with gr.Row():
                    json_config  = gr.Json( label="Config", value=None )

            inputs  = [base_model_dropdown, motion_module_dropdown, 
                       prompt_textbox, negative_prompt_textbox, width_slider, height_slider, seed_textbox,
                       filter_type_dropdown, d_s_slider, d_t_slider, num_iters_slider,
                       speed_up_options
                       ]
            # outputs = [result_video, json_config]
            # outputs = [orig_video, freeinit_video, compare_video, json_config]
            outputs = [orig_video, freeinit_video, json_config]
            
            generate_button.click( fn=controller.animate, inputs=inputs, outputs=outputs )
                
        gr.Examples( fn=controller.animate, examples=examples, inputs=inputs, outputs=outputs, cache_examples=True)

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(max_size=20)
    demo.launch()
