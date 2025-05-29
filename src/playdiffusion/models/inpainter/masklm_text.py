import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from playdiffusion.models.inpainter.llama_nar import DiffLlama

def top_k(logits, k):
    """
    logits: B, T, codebook_size
    """
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MaskGCT(nn.Module):
    def __init__(
        self,
        vocab_text: int = 13512,   # include bos, eos
        vocab_audio: int = 10000,
        num_layers: int = 20,
        num_heads: int = 16,
        num_kv_heads: int = 16,
        embed_dim: int = 1024,
        intermediate_dim: int = 4096,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_base: float = 500000.0,
    ) -> None:
        super().__init__()

        # Parameters
        self.vocab_text = vocab_text
        self.bos_idx = vocab_text - 2
        self.eos_idx = vocab_text - 1
        self.vocab_audio = vocab_audio
        self.mask_idx = vocab_text + vocab_audio
        self.text_guidance_idx = self.mask_idx + 1
        self.pad_idx = self.text_guidance_idx + 1

        self.total_vocab_size = self.pad_idx + 1
        self.dim = embed_dim
        self.hidden_dim = intermediate_dim
        self.dim_head = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.attn_dropout = attn_dropout

        # Actual layers/parameters
        self.tok_embeddings = nn.Embedding(self.total_vocab_size, self.dim, padding_idx = self.pad_idx)

        self.LM = DiffLlama(
            num_layers = self.num_layers,
            num_heads = self.num_heads,
            num_kv_heads = self.num_kv_heads,
            embed_dim = self.dim,
            intermediate_dim = self.hidden_dim,
            max_seq_len = self.max_seq_len,
            attn_dropout = self.attn_dropout,
            norm_eps = self.norm_eps,
            rope_base = self.rope_base,
        )
        self.to_logits = nn.Linear(self.dim, self.vocab_audio, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)
            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    def get_mask_prob(self, t):
        return torch.sin(t * torch.pi / 2).to(t.device)

    def convert_audio_to_vocab(self, input_token, reverse = False):
        if not reverse:
            return input_token + self.vocab_text
        else:
            return input_token - self.vocab_text

    def generate(
        self,
        text_tokens,
        target_len,
        n_timesteps=40,
        init_temp = 1.5,
        init_diversity = 1,
        guidance = 0,
        rescale_cfg = 0.75,
        topk = 20,
        code = None,
        start_frame = None,
        end_frame = None
    ):
        """
        text_tokens: B, T
        """
        bsize = text_tokens.size(0)
        device = text_tokens.device

        text_tokens = torch.cat(
            [
                self.bos_idx * torch.ones((bsize, 1), device = device, dtype = text_tokens.dtype),
                text_tokens,
                self.eos_idx * torch.ones((bsize, 1), device = device, dtype = text_tokens.dtype)
            ],
            dim = -1
        )

        if code is not None:    # inpainting
            assert start_frame is not None
            assert end_frame is not None
            T = code.size(-1)
            if start_frame > 0:
                code_before = code[:, 0 : start_frame]
            else:
                start_frame = 0
                code_before = None

            if end_frame < T-1:
                code_after = code[:, end_frame:]
            else:
                end_frame = T - 1
                code_after = None
            target_len = end_frame - start_frame + 1
        else:       # TTS
            assert start_frame is None and end_frame is None
            code_before = None
            code_after = None


        codes = self.reverse_diffusion(
            target_len = target_len,
            text_codes = text_tokens,
            n_timesteps = n_timesteps,
            init_temp = init_temp,
            init_diversity = init_diversity,
            guidance = guidance,
            rescale_cfg = rescale_cfg,
            topk = topk,
            code_before = code_before,
            code_after = code_after
        )    # B, T

        return codes

    def reverse_diffusion(
        self,
        target_len,
        text_codes,
        n_timesteps=40,
        init_temp = 1.5,
        init_diversity = 1,
        guidance = 0,
        rescale_cfg = 0.75,
        topk = 20,
        code_before = None,
        code_after = None
    ):
        """
        text_codes: B, T
        code_before: B, T
        code_after: B, T
        """
        device = text_codes.device

        bsize = text_codes.size(0)
        text_len = text_codes.size(-1)

        T_before = 0
        if code_before is not None:
            T_before = code_before.size(-1)
            code_before = self.convert_audio_to_vocab(code_before)
        if code_after is not None:
            code_after = self.convert_audio_to_vocab(code_after)

        codes = torch.full((bsize, target_len), self.mask_idx).to(device)  # (B, T)
        mask = torch.full((bsize, target_len), True).to(device)  # (B, T)

        if guidance > 0:
            text_codes_drop = torch.ones_like(text_codes) * self.text_guidance_idx
            text_codes_drop[:, 0] = self.bos_idx
            text_codes_drop[:, -1] = self.eos_idx

        h = 1.0 / n_timesteps
        t_list = [1.0 - i * h for i in range(n_timesteps)]
        t_list.append(0.0)

        for i in range(n_timesteps):
            t = t_list[i] * torch.ones(bsize).to(device)    # B
            if code_before is not None:
                codes = torch.cat([code_before, codes], dim = -1)
            if code_after is not None:
                codes = torch.cat([codes, code_after], dim = -1)

            embeds = self.tok_embeddings(torch.cat([text_codes, codes], dim = -1))    # B, T, C
            embeds = self.LM(embeds)     # B, T, C
            embeds = embeds[:, text_len:, :]      # B, T, C

            if guidance > 0:
                mask_embeds = self.tok_embeddings(torch.cat([text_codes_drop, codes], dim = -1))    # B, T, C
                mask_embeds = self.LM(mask_embeds)     # B, T, C
                mask_embeds = mask_embeds[:, text_len:, :]   # B, T, C

                pos_emb_std = embeds.std()
                embeds = embeds + guidance * (embeds - mask_embeds)
                rescale_embeds = embeds * pos_emb_std / embeds.std()
                embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

            if code_before is not None or code_after is not None:
                codes = codes[:, T_before : T_before + target_len]
                embeds = embeds[:, T_before : T_before + target_len, :]


            logits = self.to_logits(embeds)  # (B, T, codebook_size)

            annealing_scale = t_list[i]
            diversity = init_diversity * annealing_scale
            temp = init_temp * annealing_scale

            logits = top_k(logits, k = topk)    # B, T, codebook_size

            if i == n_timesteps - 1:     # last step
                sampled_ids = logits.argmax(dim=-1)    # B, T
            else:
                sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))     # B, T

            codes = torch.where(mask, self.convert_audio_to_vocab(sampled_ids), codes)

            scores = logits.softmax(dim=-1)    # B, T, codebook_size
            scores = scores.gather(2, rearrange(sampled_ids, "b n -> b n 1"))
            scores = rearrange(scores, "b n 1 -> b n")     # B, T

            scores = diversity * gumbel_noise(scores) + log(scores)
            scores = -1 * scores.float()

            next_t = t_list[i + 1] * torch.ones(bsize).to(device)

            next_mask_num = (self.get_mask_prob(next_t) * target_len).long()[0].item()

            if next_mask_num == 0:
                break
            scores = scores.masked_fill(
                ~mask, -torch.finfo(scores.dtype).max
            )  # B, T

            mask_indices = scores.topk(next_mask_num, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
                1, mask_indices, True
            )   # B, T
            codes = codes.masked_fill(mask, self.mask_idx)

        if code_before is not None:
            codes = torch.cat([code_before, codes], dim = -1)
        if code_after is not None:
            codes = torch.cat([codes, code_after], dim = -1)

        codes = self.convert_audio_to_vocab(codes, reverse = True)

        return codes



    def forward(
        self,
        codes: torch.Tensor,
        targets: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0,
        emb_scale: Optional[float] = None,
        monitor: Optional[bool] = False
    ):
        """
        codes: B, T
        targets: B, T
        """
        code_embs = self.tok_embeddings(codes)
        if emb_scale is not None:
            code_embs = emb_scale * code_embs + (1.0 - emb_scale) * code_embs.detach()

        code_embs = self.LM(
            code_embs,
            attn_mask,
        )    # B, T, C

        logits = self.to_logits(code_embs).transpose(1,2).float()     # B, C, T

        output_dict = {}

        loss = F.cross_entropy(
            input = logits,
            target = targets,     # B, T
            label_smoothing = label_smoothing,
            ignore_index = self.pad_idx
        )

        #output_dict['loss'] = loss.item()

        if monitor:
            with torch.no_grad():
                mask = (codes == self.mask_idx)
                num_samples = mask.sum()

                acc1 = (torch.argmax(logits, dim = 1) == targets).float()    # B, T
                acc1 = 100 * (acc1 * mask).sum() / num_samples

                sids = torch.argsort(logits, dim = 1, descending = True)[:, :5, :] # B, 5, T
                acc5 = (targets.unsqueeze(dim = 1) == sids).float().sum(1)   # B, T
                acc5 = 100 * (acc5 * mask).sum() / num_samples

                sids = torch.argsort(logits, dim = 1, descending = True)[:, :10, :] # B, 10, T
                acc10 = (targets.unsqueeze(dim = 1) == sids).float().sum(1)   # B, T
                acc10 = 100 * (acc10 * mask).sum() / num_samples

                output_dict['Acc1'] = acc1.item()
                output_dict['Acc5'] = acc5.item()
                output_dict['Acc10'] = acc10.item()

        return loss, output_dict


def load_maskgct_inpainter(checkpoint: str, device: str = 'cuda'):
    ckpt = torch.load(checkpoint)
    h = ckpt['config']
    state_dict = ckpt['model']
    model = MaskGCT(
        vocab_text=h["vocab_size"],  # include bos, eos
        vocab_audio=h["codebook_size"],
        num_layers=h["num_layers_lm"],
        num_heads=h["num_heads"],
        num_kv_heads=h["num_kv_heads"],
        embed_dim=h["hidden_size"],
        intermediate_dim=h["hidden_size"] * 4,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=50000.0,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model