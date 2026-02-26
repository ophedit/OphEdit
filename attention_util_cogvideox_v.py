
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Tuple
import abc

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
        
    def between_steps(self):
        return
        
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError


class AttentionStore(AttentionControl):
    """Stores complete Value tensors during DDIM inversion.
    
    Only stores for last n_steps_store steps and last m_blocks_store blocks to save memory.
    """
    
    def __init__(self, num_inference_steps: int = 50, n_steps_store: int = 10, 
                 m_blocks_store: int = 8, total_blocks: int = 30):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}  # Key: step, Value: Dict of layer storages
        
        # Storage parameters
        self.num_inference_steps = num_inference_steps
        self.n_steps_store = n_steps_store  # Only store last n steps
        self.m_blocks_store = m_blocks_store  # Only store last m blocks
        self.total_blocks = total_blocks
        
        # Track timesteps for debugging
        self.timestep_at_step = {}  # step_idx -> timestep value
        
        print(f"[STORE INIT] num_inference_steps={num_inference_steps}, n_steps_store={n_steps_store}, m_blocks_store={m_blocks_store}, total_blocks={total_blocks}")
        print(f"[STORE INIT] Will store steps {num_inference_steps - n_steps_store} to {num_inference_steps - 1}")
        print(f"[STORE INIT] Will store layers {total_blocks - m_blocks_store} to {total_blocks - 1}")
        
    def get_empty_store(self):
        return {}
    
    def set_current_timestep(self, t):
        """Called from pipeline to track actual timestep values."""
        self.timestep_at_step[self.cur_step] = t
        
    def should_store(self, layer_idx: int) -> bool:
        """Check if we should store value for current step and layer.
        
        Store only for:
        - Last n_steps_store steps (high noise levels, needed for editing)
        - Last m_blocks_store blocks
        """
        # Check step condition: last n steps of inversion
        # Inversion goes 0 -> N-1, we want steps (N-n) to (N-1)
        step_ok = self.cur_step >= (self.num_inference_steps - self.n_steps_store)
        
        # Check block condition: last m blocks
        block_ok = layer_idx >= (self.total_blocks - self.m_blocks_store)
        
        return step_ok and block_ok
        
    def step_callback(self, x_t):
        if self.step_store:  # Only store if we have data
            self.attention_store[self.cur_step] = self.step_store
            if self.cur_step == self.num_inference_steps - self.n_steps_store:
                print(f"[STORE] Started storing at step {self.cur_step}")
        
        
        self.step_store = self.get_empty_store()
        self.cur_step += 1
        self.cur_att_layer = 0
        return x_t
        
    def between_steps(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        
    def __call__(self, value_tensor_complete, layer_idx):
        """Store complete Value tensor (text + video) only for last n steps and last m blocks."""
        if self.should_store(layer_idx):
            self.step_store[layer_idx] = {
                "value_complete": value_tensor_complete.detach().cpu()
            }
            # Debug: print first time storing for this step
            first_store_layer = self.total_blocks - self.m_blocks_store
            if layer_idx == first_store_layer and self.cur_step >= (self.num_inference_steps - self.n_steps_store):
                t = self.timestep_at_step.get(self.cur_step, "unknown")
                print(f"[STORE] Step {self.cur_step} (t={t}), storing layer {layer_idx}, shape={value_tensor_complete.shape}")
        
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    def print_summary(self):
        """Print summary of what was stored."""
        print(f"\n[STORE SUMMARY] Stored {len(self.attention_store)} steps")
        print(f"[STORE SUMMARY] Stored step indices: {sorted(self.attention_store.keys())}")
        for step_idx in sorted(self.attention_store.keys())[:3]:
            t = self.timestep_at_step.get(step_idx, "unknown")
            layers = list(self.attention_store[step_idx].keys())
            print(f"[STORE SUMMARY] Step {step_idx} (t={t}): layers {layers}")


class AttentionReplace(AttentionControl):
    """Injects stored Value tensors during editing for first n timesteps and last m blocks."""
    
    def __init__(self, 
                 prompts,
                 num_steps: int,
                 source_attention_store: AttentionStore,
                 tokenizer=None,
                 n_timesteps_inject: int = 10,  # Inject value for first n timesteps
                 m_last_blocks_inject: int = 5,  # Inject value only in last m blocks
                 total_blocks: int = 30):  # Total number of attention blocks (default 30)
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_steps = num_steps
        self.source_attention_store = source_attention_store
        self.tokenizer = tokenizer
        self.n_timesteps_inject = n_timesteps_inject  # First n timesteps for value injection
        self.m_last_blocks_inject = m_last_blocks_inject  # Last m blocks for value injection
        self.total_blocks = total_blocks  # Total attention blocks in transformer
        
        # Track timesteps for debugging
        self.timestep_at_step = {}  # step_idx -> timestep value
        self.injection_count = 0
        self.miss_count = 0
        
        print(f"\n[REPLACE INIT] num_steps={num_steps}, n_timesteps_inject={n_timesteps_inject}, m_last_blocks_inject={m_last_blocks_inject}")
        print(f"[REPLACE INIT] Will inject for editing steps 0 to {n_timesteps_inject - 1}")
        print(f"[REPLACE INIT] Will inject for layers {total_blocks - m_last_blocks_inject} to {total_blocks - 1}")
        print(f"[REPLACE INIT] Source store has {len(source_attention_store.attention_store)} stored steps")
        print(f"[REPLACE INIT] Source stored step indices: {sorted(source_attention_store.attention_store.keys())}")
    
    def set_current_timestep(self, t):
        """Called from pipeline to track actual timestep values."""
        self.timestep_at_step[self.cur_step] = t
        
    def step_callback(self, x_t):
        # Print summary when exiting injection zone
        if self.cur_step == self.n_timesteps_inject:
            print(f"\n[INJECT SUMMARY] After {self.n_timesteps_inject} steps: {self.injection_count} injections, {self.miss_count} misses")
        self.cur_step += 1
        self.cur_att_layer = 0
        return x_t
    
    def should_inject_value(self, layer_idx: int) -> bool:
        """Check if value injection should happen for current step and layer.
        
        Inject complete value only for:
        - First n_timesteps_inject timesteps
        - Last m_last_blocks_inject blocks (layers)
        """
        # Check timestep condition: first n timesteps
        timestep_ok = self.cur_step < self.n_timesteps_inject
        
        # Check block condition: last m blocks
        # If total_blocks=30 and m_last_blocks_inject=5, inject for layers 25,26,27,28,29
        block_ok = layer_idx >= (self.total_blocks - self.m_last_blocks_inject)
        
        return timestep_ok and block_ok

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class CogVideoXAttnProcessor:
    r"""
    Processor for implementing FateZero-style editing on CogVideoX.
    Only injects complete Value tensors (no cross-attention manipulation).
    """

    def __init__(self, controller: AttentionControl = None, layer_idx: int = 0):
        self.controller = controller
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn,  # The Attention Module
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        text_seq_length = encoder_hidden_states.size(1)

        # Concatenate Text + Video
        # [Batch, Text_Seq, Dim] + [Batch, Video_Seq, Dim] -> [Batch, Total_Seq, Dim]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # Shape: [Batch, Heads, Seq_Len, Dim]
        
        # Normalization (CogVideoX specific)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE to video portion only
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # --- Store Complete Value (if Inversion) ---
        if isinstance(self.controller, AttentionStore):
            # Store COMPLETE Value (text + video parts)
            # Shape: [Batch, Heads, Total_Seq (text + video), Dim]
            self.controller(value, self.layer_idx)

        # --- Value Injection (for first n timesteps and last m blocks) ---
        # Only inject on conditional part during CFG (batch index 1), not unconditional (batch index 0)
        if isinstance(self.controller, AttentionReplace) and self.controller.should_inject_value(self.layer_idx):
            try:
                # Map editing step to inversion step
                # Editing step 0 (noisy) -> Inversion's last step (also noisy)
                # Inversion stored steps: (num_steps - n_store) to (num_steps - 1)
                store = self.controller.source_attention_store
                num_inv_steps = store.num_inference_steps
                
                # Editing step i corresponds to inversion step (num_inv_steps - 1 - i)
                source_step_idx = num_inv_steps - 1 - self.controller.cur_step
                
                # Debug: print once per step for first matching layer
                first_inject_layer = self.controller.total_blocks - self.controller.m_last_blocks_inject
                if self.layer_idx == first_inject_layer:
                    print(f"[DEBUG] Editing step {self.controller.cur_step} -> Looking for inversion step {source_step_idx}")
                    print(f"[DEBUG] Stored steps: {list(store.attention_store.keys())[:5]}...{list(store.attention_store.keys())[-5:]}")
                
                if source_step_idx in store.attention_store and self.layer_idx in store.attention_store[source_step_idx]:
                    source_layer_data = store.attention_store[source_step_idx][self.layer_idx]
                    source_value_complete = source_layer_data["value_complete"].to(value.device).to(value.dtype)
                    
                    self.controller.injection_count += 1
                    
                    if self.layer_idx == first_inject_layer:
                        print(f"[INJECT] Step {self.controller.cur_step}, Layer {self.layer_idx}: Injecting value from inv step {source_step_idx}")
                        print(f"[INJECT] Source shape: {source_value_complete.shape}, Target shape: {value.shape}")
                        print(f"[INJECT] Source value stats: min={source_value_complete.min().item():.4f}, max={source_value_complete.max().item():.4f}")
                        print(f"[INJECT] Target value BEFORE: min={value.min().item():.4f}, max={value.max().item():.4f}")
                    
                    # Adaptive Normalization to prevent domain mismatch
                    def adaptive_norm(src_v, tgt_v):
                        src_mean = src_v.mean(dim=-2, keepdim=True)
                        src_std = src_v.std(dim=-2, keepdim=True) + 1e-5
                        tgt_mean = tgt_v.mean(dim=-2, keepdim=True)
                        tgt_std = tgt_v.std(dim=-2, keepdim=True) + 1e-5
                        return ((src_v - src_mean) / src_std) * tgt_std + tgt_mean

                    # During CFG, batch_size=2: [uncond, cond]
                    # source_value_complete has batch_size=1 (from inversion without CFG)
                    if batch_size == 2:
                        # CFG mode: only replace conditional (index 1)
                        value[1] = adaptive_norm(source_value_complete[0], value[1])
                    elif batch_size == 1:
                        # Non-CFG mode: replace all
                        # value and source_value_complete both have batch_size=1
                        value = adaptive_norm(source_value_complete, value)
                    else:
                        value[-1] = adaptive_norm(source_value_complete[0], value[-1])
                    
                    if self.layer_idx == first_inject_layer:
                        print(f"[INJECT] Target value AFTER: min={value.min().item():.4f}, max={value.max().item():.4f}")
                else:
                    self.controller.miss_count += 1
                    if self.layer_idx == first_inject_layer:
                        print(f"[MISS] Step {self.controller.cur_step}: source_step_idx {source_step_idx} not found or layer {self.layer_idx} not stored")
                        print(f"[MISS] Available step indices: {sorted(store.attention_store.keys())}")
                        if source_step_idx in store.attention_store:
                            print(f"[MISS] Available layers for step {source_step_idx}: {list(store.attention_store[source_step_idx].keys())}")
            except Exception as e:
                import traceback
                print(f"[Error] Value injection failed: {e}")
                traceback.print_exc()

        # --- Attention Calculation ---
        scale = 1 / (head_dim ** 0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute Output
        hidden_states = torch.matmul(attn_weights, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # Linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Split back to [Text, Video] and return [Video] (as per diffusers convention)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        
        return hidden_states, encoder_hidden_states
