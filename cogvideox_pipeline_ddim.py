
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import copy
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from tqdm import tqdm
from diffusers.utils import logging

from ..prompt_attention.attention_util_cogvideox_v import (
    AttentionStore, 
    AttentionReplace, 
    CogVideoXAttnProcessor
)

logger = logging.get_logger(__name__)

class CogVideoXEditPipeline(CogVideoXPipeline):
    
    def register_attention_control(self, controller):

        attn_procs = {}
        for name in self.transformer.attn_processors.keys():
            # Create a new processor instance for each layer, sharing the same controller
            # We assign a layer_idx to help the controller keep track if needed (though our simple store extends lists)
            
            # Note: The order of keys in attn_processors is usually consistent.
            # But to be safe, we might just rely on the controller's internal counter 
            # if we didn't implement layer_idx logic perfectly. 
            # In our util, we passed layer_idx. Here we need to count.
            attn_procs[name] = CogVideoXAttnProcessor(controller=controller, layer_idx=0) 
        
        self.transformer.set_attn_processor(attn_procs)
        
        # Now update layer_idx
        # We need to traverse modules to ensure we index them in the same order as execution
        i = 0
        for name, module in self.transformer.named_modules():
             if hasattr(module, "processor") and isinstance(module.processor, CogVideoXAttnProcessor):
                 module.processor.layer_idx = i
                 i += 1
        
        self.controller = controller

    @torch.no_grad()
    def invert(
        self,
        prompt: Union[str, List[str]],
        video_latents: torch.Tensor, # Expected shape: [Batch, Channels, Frames, Height, Width]
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        n_steps_store: int = 10,  # Only store last n steps to save memory
        m_blocks_store: int = 8,  # Only store last m blocks to save memory
        device = None,
    ):
        r"""
        Inverts a video to noise using DDIM inversion.
        Returns:
            inverted_latents: The noisy latents at timestep T
            attention_store: The recorded attention maps and values
        """
        device = device or self._execution_device
        self.transformer.to(device) # Ensure model is on correct device
        
        # Count total attention blocks
        total_blocks = len([n for n in self.transformer.attn_processors.keys()])
        
        # 0. Prepare controller - only store last n steps and last m blocks
        attention_store = AttentionStore(
            num_inference_steps=num_inference_steps,
            n_steps_store=n_steps_store,
            m_blocks_store=m_blocks_store,
            total_blocks=total_blocks
        )
        
        self.register_attention_control(attention_store)
        
        # 1. Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False # Inversion usually done unconditionally or with null text? 
            # FateZero Paper: "We invert the input video... using the source prompt" 
            # Inversion equation typically uses NO CFG (or CFG=1).
            # So we use prompt_embeds only.
        )
        # Note: CogVideoX expects prompt_embeds to be passed as encoder_hidden_states
        
        # 2. Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # For inversion, we need to go from t=0 -> t=T (clean to noisy).
        # scheduler.timesteps are T -> 0, so we flip them.
        timesteps = self.scheduler.timesteps.flip(0)
        
        print(f"Scheduler timesteps (first 5): {timesteps[:5].tolist()}")
        print(f"Scheduler timesteps (last 5): {timesteps[-5:].tolist()}")
        print(f"alphas_cumprod[0] = {self.scheduler.alphas_cumprod[0].item():.6f}")
        print(f"alphas_cumprod[-1] = {self.scheduler.alphas_cumprod[-1].item():.6f}")
        
        # 3. Latents
        # Assuming video_latents is already encoded VAE latents.
        latents = video_latents.to(device).to(prompt_embeds.dtype)
        
        print(f"Initial latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, std={latents.std().item():.4f}")
        
        # 4. Inversion Loop
        print("Running DDIM Inversion...")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # Convert timestep to int for proper indexing
                t_int = int(t.item())
                
                # Check rotary requirements
                if self.transformer.config.use_rotary_positional_embeddings:
                    # Latents are [B, F, C, H, W] (confirmed from training code)
                    batch_size, num_frames, num_channels, height, width = latents.shape
                    image_rotary_emb = self._prepare_rotary_positional_embeddings(
                        height * self.vae_scale_factor_spatial, 
                        width * self.vae_scale_factor_spatial, 
                        num_frames, 
                        device
                    )
                else:
                    image_rotary_emb = None
                
                # Expand timestep to batch size
                timestep = t.unsqueeze(0).expand(latents.shape[0])

                # Forward pass - predict noise
                noise_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False
                )[0]
                
                # Debug: check model output
                if i == 0 or i == len(timesteps) - 1:
                    print(f"Step {i}, t={t_int}: model_output stats: min={noise_pred.min().item():.4f}, max={noise_pred.max().item():.4f}")
                
                # Get alpha values using integer indexing
                alpha_prod_t = self.scheduler.alphas_cumprod[t_int]
                
                # Next timestep
                if i < len(timesteps) - 1:
                    next_t_int = int(timesteps[i+1].item())
                    alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t_int]
                else:
                    # At the final step, we're at max noise level
                    alpha_prod_t_next = self.scheduler.alphas_cumprod[-1]
                
                if i == 0 or i == len(timesteps) - 1:
                    print(f"Step {i}: alpha_t={alpha_prod_t.item():.6f}, alpha_next={alpha_prod_t_next.item():.6f}")
                
                # Handle v-prediction: CogVideoX uses v-prediction
                # v = sqrt(alpha) * epsilon - sqrt(1-alpha) * x0
                # 
                # To get epsilon and x0 from v:
                # epsilon = sqrt(alpha) * v + sqrt(1-alpha) * x_t
                # x0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v
                
                sqrt_alpha_t = alpha_prod_t ** 0.5
                sqrt_one_minus_alpha_t = (1 - alpha_prod_t) ** 0.5
                
                # Check prediction type
                prediction_type = self.scheduler.config.prediction_type
                
                if prediction_type == "v_prediction":
                    # For v-prediction:
                    # pred_x0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
                    # pred_eps = sqrt(alpha_t) * v + sqrt(1-alpha_t) * x_t
                    pred_x0 = sqrt_alpha_t * latents - sqrt_one_minus_alpha_t * noise_pred
                    pred_eps = sqrt_alpha_t * noise_pred + sqrt_one_minus_alpha_t * latents
                elif prediction_type == "epsilon":
                    # For epsilon-prediction:
                    # pred_x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
                    sqrt_alpha_t_clamped = torch.clamp(sqrt_alpha_t, min=1e-8)
                    pred_x0 = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t_clamped
                    pred_eps = noise_pred
                else:
                    raise ValueError(f"Unknown prediction type: {prediction_type}")
                
                # DDIM Inversion: compute next (noisier) latent
                # z_{t+1} = sqrt(alpha_{t+1}) * pred_x0 + sqrt(1-alpha_{t+1}) * pred_eps
                sqrt_alpha_t_next = alpha_prod_t_next ** 0.5
                sqrt_one_minus_alpha_t_next = (1 - alpha_prod_t_next) ** 0.5
                
                latents = sqrt_alpha_t_next * pred_x0 + sqrt_one_minus_alpha_t_next * pred_eps
                
                # Check for NaN/Inf
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"NaN/Inf detected at step {i}, t={t_int}!")
                    print(f"  alpha_t: {alpha_prod_t.item():.6f}, alpha_next: {alpha_prod_t_next.item():.6f}")
                    print(f"  pred_x0 stats: min={pred_x0.min().item():.4f}, max={pred_x0.max().item():.4f}")
                    print(f"  pred_eps stats: min={pred_eps.min().item():.4f}, max={pred_eps.max().item():.4f}")
                    break
                
                if i == 0 or i == len(timesteps) - 1:
                    print(f"Step {i}: latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
                
                # Update controller
                self.controller.set_current_timestep(t_int)  # Track timestep
                self.controller.step_callback(latents)
                progress_bar.update()
        
        print(f"Final inverted latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, std={latents.std().item():.4f}")
        
        # Print storage summary
        attention_store.print_summary()

        return latents, attention_store

    @torch.no_grad()
    def __call__( # Editing
        self,
        prompt: Union[str, List[str]] = None,
        latents: torch.Tensor = None, # Start from inverted latents
        attention_store: AttentionStore = None, # Source attention
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        n_timesteps_inject: int = 10,  # Inject complete value for first n timesteps
        m_last_blocks_inject: int = 5,  # Inject complete value only in last m blocks (out of 30)
        **kwargs
    ):
        # Count total attention blocks in transformer (default 30 for CogVideoX)
        total_blocks = len([n for n in self.transformer.attn_processors.keys()])
        
        # 0. Setup Controller - only Value injection, no cross/self attention manipulation
        controller = AttentionReplace(
            prompts=[prompt], 
            num_steps=num_inference_steps,
            source_attention_store=attention_store,
            tokenizer=self.tokenizer,
            n_timesteps_inject=n_timesteps_inject,
            m_last_blocks_inject=m_last_blocks_inject,
            total_blocks=total_blocks
        )
        self.register_attention_control(controller)
        
        # 1. Run Standard Pipeline Loop
        # We can just call super().__call__, passing latents!
        # Because we already replaced the attention processors.
        
        # NOTE: CogVideoXPipeline.__call__ handles loop, and calls self.transformer(...)
        # which calls our processors.
        # However, we need to ensure the callbacks (step_callback) are called.
        
        # The standard pipeline has `callback_on_step_end`.
        # We can pass a callback function to it.
        # 
        # IMPORTANT: callback_on_step_end is called AFTER the step, so:
        # - We call step_callback to increment cur_step for next iteration
        # - The controller.cur_step is used during attention forward to look up values
        # - Initial cur_step = 0 is correct for the first step
        
        def step_callback_fn(pipeline, step, t, callback_kwargs):
            # Debug: print timestep mapping at key points
            t_val = int(t.item()) if hasattr(t, 'item') else int(t)
            if step < 5 or step >= num_inference_steps - 5:
                store = attention_store
                source_step_idx = store.num_inference_steps - 1 - step
                print(f"[EDIT STEP DONE] step={step}, t={t_val}, was looking for inv step {source_step_idx}")
            
            # Now increment for next step
            self.controller.step_callback(None)
            return callback_kwargs
            # CogVideoXPipeline.py source shows: callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            # And it expects dict return to update latents etc.
        
        print(f"\\n[EDITING] Starting edit with {num_inference_steps} steps, n_inject={n_timesteps_inject}, m_blocks={m_last_blocks_inject}")
        print(f"[EDITING] Controller cur_step starts at {controller.cur_step}")
        
        return super().__call__(
            prompt=prompt,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback_on_step_end=step_callback_fn,
            **kwargs
        )
