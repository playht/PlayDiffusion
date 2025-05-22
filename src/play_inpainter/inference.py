from dataclasses import dataclass
from typing import List, Optional

from play_inpainter.models.model_manager import InpainterModelManager
from play_inpainter.pydantic_models.models import InpainterInput
from play_inpainter.utils.audio_utils import Timer, get_vocoder_embedding, load_audio
from play_inpainter.utils.save_audio import make_16bit_pcm


@dataclass
class TextDiffChunk:
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int
    ref_buf_start: int
    ref_buf_end: int
    hyp_buf_start: int
    hyp_buf_end: int


@dataclass
class InpainterChunk:
    import torch

    start_frame: Optional[int]
    end_frame: Optional[int]
    n_frames: int
    buf_start_frame: Optional[int]
    buf_end_frame: Optional[int]
    text_tokens: torch.Tensor
    start_silence_frames: int
    end_silence_frames: int


class Inpainter():
    def __init__(self, device: str = "cuda"):
        import nltk
        import torch
        from g2p_en import G2p

        self.device = torch.device(device)

        self.preset = self.load_preset()
        self.mm = InpainterModelManager(self.preset, self.device)

        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        self.g2p = G2p()

        self.frame_rate = 50
        self.default_audio_token_phoneme_ratio = (
            self.frame_rate / 11
        )  # avg en speaker speaks 10-12 phonemes/sec
        self.max_audio_frames = 750

        self.timer = Timer()

    def load_preset(
        self,
        vocoder_path: str = "s3://play-fal/ldm-models/v090_g_01105000",
        tokenizer_path: str = "s3://play-fal/v3-models/tokenizer-multi_bpe16384_merged_extended_58M.json",
        speech_tokenizer_path: str = "s3://play-fal/ldm-models/xlsr2_1b_v2_custom.pt",
        kmeans_layer_path: str = "s3://play-fal/ldm-models/kmeans_10k.npy",
        inpainter_path: str = "s3://play-fal/ldm-models/last_250k_fixed.pkl",
    ) -> dict:
        from play_inpainter.utils.preset_util import get_all_checkpoints_from_preset

        preset = dict(
            vocoder = dict(
                checkpoint = vocoder_path,
                kmeans_layer_checkpoint = kmeans_layer_path,
            ),
            tokenizer = dict(
                vocab_file = tokenizer_path
            ),
            speech_tokenizer = dict(
                checkpoint = speech_tokenizer_path,
                kmeans_layer_checkpoint = kmeans_layer_path
            ),
            mel = dict(
                sample_rate = 16000
            ),
            voice_encoder = dict(
                spec_dim = 100,
                embedding_dim = 1024,
                attn_blocks = 6,
            ),
            inpainter = dict(
                checkpoint = inpainter_path
            ),
        )

        preset = get_all_checkpoints_from_preset(preset)

        return preset

    def g2p_stripped(self, text: str) -> List[str]:
        non_phonetic = [" ", "", ",", "'"]
        phonemes = self.g2p(text)
        phonemes = [p for p in phonemes if p not in non_phonetic]
        return phonemes

    def inpaint(self, input: InpainterInput):
        import jiwer
        import numpy as np
        import torch
        import torchaudio.functional as F

        self.timer.reset()

        print(f"Inpainter input text: {input.input_text}")
        print(f"Inpainter output text: {input.output_text}")
        print("Inpainter input word times:")
        for i, word in enumerate(input.input_word_times):
            print(f"    {i}: {word}")

        # determine diffs between input and output text
        text_align = jiwer.process_words(input.input_text, input.output_text)

        # use the input audio itself for the vocoder embedding
        vocoder_emb = get_vocoder_embedding(input.audio, self.mm)
        self.timer("Get vocoder embedding")

        # extract xlsr audio tokens
        input_wav, sr = load_audio(input.audio, self.device)
        self.timer("Load audio")
        resampled_wav = F.resample(
            input_wav, orig_freq=sr, new_freq=self.mm.speech_tokenizer_sample_rate
        )
        print(f"Resampled wav: {resampled_wav.shape}")
        self.timer("Resample")
        with torch.inference_mode():
            input_audio_tokens = self.mm.speech_tokenizer.waveform_to_units(
                resampled_wav.squeeze()
            )
        print(f"Input audio tokens: {input_audio_tokens.shape}")
        self.timer("Speech tokenizer")

        # determine audio-token-to-phoneme ratio (excluding long silences)
        speech_time = 0.0
        break_spacing = 1.0
        last_word_end = -break_spacing
        words_to_phonemize = []
        for i, word in enumerate(input.input_word_times):
            words_to_phonemize.append(word["word"])
            if word["start"] - last_word_end < break_spacing:
                speech_time += word["end"] - last_word_end
            else:
                speech_time += word["end"] - last_word_end - break_spacing
            last_word_end = word["end"]
        phonemes = self.g2p_stripped(" ".join(words_to_phonemize))
        audio_token_phoneme_ratio = 1.2 * speech_time * self.frame_rate / len(phonemes)
        print(f"Audio token to phoneme ratio: {audio_token_phoneme_ratio}")
        self.timer("Audio token to phoneme ratio")

        # merge adjacent diff chunks and calculate static buffer to pass to inpainter
        merged_chunks = []  # type: ignore
        # if no silence, how many extra words to inpaint on each side of actual change
        dynamic_word_buffer = 1
        # if there is silence, how much time to inpaint on each side of actual change
        dynamic_time_buffer = 0.25
        # how many extra words to show the model on each side of the inpainted region
        static_word_buffer = 5
        last_chunk_end = -1
        for chunk in text_align.alignments[0]:
            if chunk.type != "equal":
                ref_start = chunk.ref_start_idx
                hyp_start = chunk.hyp_start_idx
                ref_end = chunk.ref_end_idx - 1
                hyp_end = chunk.hyp_end_idx - 1

                # add dynamic word buffer to the start/end of the ref/hyp
                # unless there is a long silence between the chunks
                ref_start_time = (
                    input.input_word_times[ref_start]["start"]
                    if ref_start < len(input.input_word_times)
                    else None
                )
                prev_word_end_time = (
                    input.input_word_times[ref_start - 1]["end"]
                    if ref_start > 0
                    else None
                )
                has_begin_silence = (
                    ref_start_time is not None
                    and prev_word_end_time is not None
                    and ref_start_time - prev_word_end_time > dynamic_time_buffer * 2
                )
                if ref_start - dynamic_word_buffer < 0:
                    ref_start = 0
                elif not has_begin_silence:
                    ref_start = ref_start - dynamic_word_buffer
                if hyp_start - dynamic_word_buffer < 0:
                    hyp_start = 0
                elif not has_begin_silence:
                    hyp_start = hyp_start - dynamic_word_buffer
                ref_end_time = (
                    input.input_word_times[ref_end]["end"]
                    if ref_end < len(input.input_word_times)
                    else None
                )
                next_word_start_time = (
                    input.input_word_times[ref_end + 1]["start"]
                    if ref_end < len(input.input_word_times) - 1
                    else None
                )
                has_end_silence = (
                    ref_end_time is not None
                    and next_word_start_time is not None
                    and next_word_start_time - ref_end_time > dynamic_time_buffer * 2
                )
                if ref_end + dynamic_word_buffer >= len(input.input_word_times):
                    ref_end = len(input.input_word_times) - 1
                elif not has_end_silence:
                    ref_end = ref_end + dynamic_word_buffer
                if hyp_end + dynamic_word_buffer >= len(text_align.hypotheses[0]):
                    hyp_end = len(text_align.hypotheses[0]) - 1
                elif not has_end_silence:
                    hyp_end = hyp_end + dynamic_word_buffer

                # add static word buffer to the start/end of the ref/hyp
                if ref_start - static_word_buffer < 0:
                    ref_buf_start = 0
                else:
                    ref_buf_start = ref_start - static_word_buffer
                if hyp_start - static_word_buffer < 0:
                    hyp_buf_start = 0
                else:
                    hyp_buf_start = hyp_start - static_word_buffer
                if ref_end + static_word_buffer >= len(input.input_word_times):
                    ref_buf_end = len(input.input_word_times) - 1
                else:
                    ref_buf_end = ref_end + static_word_buffer
                if hyp_end + static_word_buffer >= len(text_align.hypotheses[0]):
                    hyp_buf_end = len(text_align.hypotheses[0]) - 1
                else:
                    hyp_buf_end = hyp_end + static_word_buffer

                # if the chunk is adjacent to the previous chunk, merge them
                # but don't merge if there is a long silence between the chunks
                if (
                    input.input_word_times[ref_start].get("start") is not None
                    and last_chunk_end >= 0
                    and last_chunk_end < len(input.input_word_times) - 1
                    and input.input_word_times[last_chunk_end].get("end") is not None
                ):
                    has_silence = (
                        input.input_word_times[ref_start]["start"]
                        - input.input_word_times[last_chunk_end]["end"]
                        > dynamic_time_buffer * 2
                    )
                else:
                    has_silence = False
                if ref_start <= last_chunk_end and not has_silence:
                    merged_chunks[-1].ref_end = ref_end + 1
                    merged_chunks[-1].ref_buf_end = ref_buf_end + 1
                    merged_chunks[-1].hyp_end = hyp_end + 1
                    merged_chunks[-1].hyp_buf_end = hyp_buf_end + 1
                else:
                    diff_chunk = TextDiffChunk(
                        ref_start=ref_start,
                        ref_end=ref_end + 1,
                        hyp_start=hyp_start,
                        hyp_end=hyp_end + 1,
                        ref_buf_start=ref_buf_start,
                        ref_buf_end=ref_buf_end + 1,
                        hyp_buf_start=hyp_buf_start,
                        hyp_buf_end=hyp_buf_end + 1,
                    )
                    merged_chunks.append(diff_chunk)
                last_chunk_end = ref_end
        self.timer("Prepare word times and diff chunks")

        # iterate over the diffs, calculating inputs for the inpainter
        diffs = []
        for chunk in merged_chunks:
            ref_start = chunk.ref_start
            hyp_start = chunk.hyp_start
            ref_end = chunk.ref_end - 1
            hyp_end = chunk.hyp_end - 1
            ref_buf_start = chunk.ref_buf_start
            hyp_buf_start = chunk.hyp_buf_start
            ref_buf_end = chunk.ref_buf_end - 1
            hyp_buf_end = chunk.hyp_buf_end - 1

            # compute the start/end frames of the inpainted region and the static buffer
            if ref_start == 0:
                start_frame = 0
                start_silence_frames = 0
            else:
                start_frame = int(input.input_word_times[ref_start]["start"] * self.frame_rate)
                if (
                    input.input_word_times[ref_start]["start"]
                    - input.input_word_times[ref_start - 1]["end"]
                    > dynamic_time_buffer * 2
                ):
                    silence_seconds = (
                        input.input_word_times[ref_start]["start"]
                        - input.input_word_times[ref_start - 1]["end"]
                    ) / 2
                    start_silence_frames = int(silence_seconds * self.frame_rate)
                else:
                    start_silence_frames = 0
            if ref_end == len(input.input_word_times) - 1:
                end_frame = input_audio_tokens.shape[-1]
                end_silence_frames = 0
            else:
                end_frame = int(input.input_word_times[ref_end]["end"] * self.frame_rate)
                if (
                    input.input_word_times[ref_end + 1]["start"]
                    - input.input_word_times[ref_end]["end"]
                    > dynamic_time_buffer * 2
                ):
                    silence_seconds = (
                        input.input_word_times[ref_end + 1]["start"]
                        - input.input_word_times[ref_end]["end"]
                    ) / 2
                    end_silence_frames = int(silence_seconds * self.frame_rate)
                else:
                    end_silence_frames = 0
            if ref_buf_start == 0:
                buf_start_frame = 0
            else:
                buf_start_frame = int(
                    input.input_word_times[ref_buf_start]["start"] * self.frame_rate
                )
            if ref_buf_end == len(input.input_word_times) - 1:
                buf_end_frame = input_audio_tokens.shape[-1]
            else:
                buf_end_frame = int(
                    input.input_word_times[ref_buf_end]["end"] * self.frame_rate
                )

            # keep track of the text at each step, to tokenize and pass to the inpainter
            ref_text = text_align.references[0][ref_start : ref_end + 1]
            hyp_text = text_align.hypotheses[0][hyp_start : hyp_end + 1]
            print(f"Ref text: {ref_text}")
            print(f"Hyp text: {hyp_text}")
            ref_text_buf = text_align.references[0][ref_buf_start : ref_buf_end + 1]
            hyp_text_buf = text_align.hypotheses[0][hyp_buf_start : hyp_buf_end + 1]
            print(f"Ref text (buffered): {ref_text_buf}")
            print(f"Hyp text (buffered): {hyp_text_buf}")
            text_to_submit = (
                ref_text_buf[: ref_start - ref_buf_start]
                + hyp_text
                + ref_text_buf[ref_end + 1 - ref_buf_start :]
            )
            print(f"Text to submit: {text_to_submit}")

            # determine number of phonemes and estimate number of frames for the output
            hyp_phonemes = self.g2p_stripped(hyp_text)
            n_frames = int(len(hyp_phonemes) * audio_token_phoneme_ratio)

            # crudely split the diff into multiple chunks if it's too long
            if n_frames > self.max_audio_frames:
                n_subdiffs = 1 + n_frames // self.max_audio_frames
                sub_words = np.array_split(text_to_submit, n_subdiffs)
                for i in range(n_subdiffs):
                    if i == 0:
                        hyp_text = " ".join(sub_words[i][static_word_buffer:])
                        sub_start_frame = start_frame
                        sub_start_silence_frames = start_silence_frames
                        sub_end_frame = None
                        sub_end_silence_frames = 0
                        sub_buf_start_frame = buf_start_frame
                        sub_buf_end_frame = None
                    elif i == n_subdiffs - 1:
                        hyp_text = " ".join(sub_words[i][:-static_word_buffer])
                        sub_start_frame = None
                        sub_start_silence_frames = 0
                        sub_end_frame = end_frame
                        sub_end_silence_frames = end_silence_frames
                        sub_buf_start_frame = None
                        sub_buf_end_frame = buf_end_frame
                    else:
                        hyp_text = " ".join(sub_words[i])
                        sub_start_frame = None
                        sub_start_silence_frames = 0
                        sub_end_frame = None
                        sub_end_silence_frames = 0
                        sub_buf_start_frame = None
                        sub_buf_end_frame = None
                    hyp_phonemes = self.g2p_stripped(hyp_text)
                    n_frames = int(len(hyp_phonemes) * audio_token_phoneme_ratio)
                    sub_text_to_submit = " ".join(sub_words[i])
                    sub_text_tokens = self.mm.tokenizer.encode_normalized_to_tensor(
                        sub_text_to_submit
                    )
                    diffs.append(
                        InpainterChunk(
                            sub_start_frame,
                            sub_end_frame,
                            n_frames,
                            sub_buf_start_frame,
                            sub_buf_end_frame,
                            sub_text_tokens,
                            sub_start_silence_frames,
                            sub_end_silence_frames,
                        )
                    )
            else:
                text_tokens = self.mm.tokenizer.encode_normalized_to_tensor(
                    " ".join(text_to_submit)
                )
                diffs.append(
                    InpainterChunk(
                        start_frame,
                        end_frame,
                        n_frames,
                        buf_start_frame,
                        buf_end_frame,
                        text_tokens,
                        start_silence_frames,
                        end_silence_frames,
                    )
                )
        print(f"{len(diffs)} diffs to inpaint")
        self.timer("Calculate diffs to inpaint")

        # generate inpainted audio
        with torch.no_grad():
            output_chunks = []
            last_end_frame = -1
            for chunk in diffs:
                start_frame = chunk.start_frame
                end_frame = chunk.end_frame
                n_frames = chunk.n_frames
                buf_start_frame = chunk.buf_start_frame
                buf_end_frame = chunk.buf_end_frame
                text_tokens = chunk.text_tokens
                start_silence_frames = chunk.start_silence_frames
                end_silence_frames = chunk.end_silence_frames
                start_time = (
                    start_frame / self.frame_rate if start_frame is not None else -1
                )
                end_time = end_frame / self.frame_rate if end_frame is not None else -1
                print(
                    f"Generating inpainted audio for {start_frame} ({start_time:.2f}s) \
                    to {end_frame} ({end_time:.2f}s) with {n_frames} inpainted frames \
                    ({n_frames / self.frame_rate:.2f}s), {start_silence_frames} \
                    ({start_silence_frames / self.frame_rate:.2f}s) start silence, \
                    {end_silence_frames} ({end_silence_frames / self.frame_rate:.2f}s) \
                    end silence"
                )
                print(f"Text tokens shape: {text_tokens.shape}")
                text = self.mm.tokenizer.decode_tokens_tensor(text_tokens)
                print(f"Text: {text}")

                # save the previous unchanged audio tokens
                if (
                    start_frame is not None
                    and start_frame - start_silence_frames > last_end_frame + 1
                ):
                    output_chunks.append(
                        input_audio_tokens[
                            :, last_end_frame + 1 : start_frame - start_silence_frames
                        ]
                    )
                    print(f"Unchanged audio tokens shape: {output_chunks[-1].shape}")
                if end_frame is not None:
                    last_end_frame = end_frame + end_silence_frames

                # assemble the tokens to pass to the inpainter
                # possibly including static buffer at begin and/or end
                tokens_to_cat = []
                if buf_start_frame is not None and start_frame is not None:
                    tokens_to_cat.append(
                        input_audio_tokens[
                            :, buf_start_frame : start_frame - start_silence_frames
                        ]
                    )
                    if start_silence_frames > 0:
                        tokens_to_cat.append(
                            torch.full(
                                (1, start_silence_frames),
                                8857,
                                dtype=torch.int32,
                                device=self.device,
                            )
                        )
                    inpainter_start_frame = start_frame - buf_start_frame
                else:
                    inpainter_start_frame = 0
                # make sure the region to inpaint is the correct length
                tokens_to_cat.append(
                    torch.full(
                        (1, n_frames), 8857, dtype=torch.int32, device=self.device
                    )
                )
                if end_frame is not None and buf_end_frame is not None:
                    if end_silence_frames > 0:
                        tokens_to_cat.append(
                            torch.full(
                                (1, end_silence_frames),
                                8857,
                                dtype=torch.int32,
                                device=self.device,
                            )
                        )
                    tokens_to_cat.append(
                        input_audio_tokens[
                            :, end_frame + end_silence_frames : buf_end_frame
                        ]
                    )
                    inpainter_end_frame = inpainter_start_frame + n_frames
                else:
                    inpainter_end_frame = -1
                inpaint_audio_tokens = torch.cat(tokens_to_cat, dim=1)
                if inpainter_end_frame == -1:
                    inpainter_end_frame = inpaint_audio_tokens.shape[1]

                # inpaint the desired region, with the static buffer unmodified
                print(f"Pre-inpaint audio tokens shape: {inpaint_audio_tokens.shape}")
                inpaint_audio_tokens = self.mm.inpainter.generate(
                    text_tokens=text_tokens,
                    target_len=None,
                    n_timesteps=input.num_steps,
                    init_temp=input.init_temp,
                    init_diversity=input.init_diversity,
                    guidance=input.guidance,
                    rescale_cfg=input.rescale,
                    topk=input.topk,
                    code=inpaint_audio_tokens,
                    start_frame=inpainter_start_frame,
                    end_frame=inpainter_end_frame,
                )  # 1, T
                print(f"Post-inpaint audio tokens shape: {inpaint_audio_tokens.shape}")

                # save the actual inpainted audio tokens
                inpaint_audio_tokens = inpaint_audio_tokens[
                    :,
                    inpainter_start_frame - start_silence_frames : inpainter_start_frame
                    + n_frames
                    + end_silence_frames,
                ]
                print(
                    f"Trimmed inpainted audio token shape: {inpaint_audio_tokens.shape}"
                )
                output_chunks.append(inpaint_audio_tokens)
        self.timer("Inpainter")

        # save the remaining unchanged audio tokens and concatenate all chunks
        if input_audio_tokens.shape[-1] > last_end_frame + 1:
            output_chunks.append(input_audio_tokens[:, last_end_frame + 1 :])
            print(f"Unchanged audio tokens shape: {output_chunks[-1].shape}")
        output_audio_tokens = torch.cat(output_chunks, dim=1)

        # vocode the output audio
        with torch.no_grad():
            audio_g = self.mm.vocoder(output_audio_tokens, vocoder_emb).squeeze()
        self.timer("Vocoder")

        # encode audio for output
        return make_16bit_pcm(audio_g)
