from dataclasses import dataclass
from typing import Dict, List, Optional

from playdiffusion.models.model_manager import PlayDiffusionModelManager
from playdiffusion.pydantic_models.models import InpaintInput, TTSInput, RVCInput
from playdiffusion.utils.audio_utils import Timer, get_vocoder_embedding, load_audio
from playdiffusion.utils.save_audio import make_16bit_pcm


@dataclass
class TextDiffChunk:
    # start/end of the region to inpaint (including dynamic buffer)
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int
    # start/end of the static buffer (passed to inpainter but not modified)
    ref_buf_start: int
    ref_buf_end: int
    hyp_buf_start: int
    hyp_buf_end: int
    # start/end of the *actual* diff (not including dynamic buffer)
    ref_diff_start: int
    ref_diff_end: int


@dataclass
class InpainterChunk:
    import torch

    start_frame: Optional[int]
    end_frame: Optional[int]
    n_frames: int
    buf_start_frame: Optional[int]
    buf_end_frame: Optional[int]
    text_tokens: torch.Tensor


class PlayDiffusion():
    def __init__(self, device: str = "cuda"):
        import nltk
        import torch

        self.device = torch.device(device)

        self.preset = self.load_preset()
        self.mm = PlayDiffusionModelManager(self.preset, self.device)

        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        self.frame_rate = 50
        # avg en speaker speaks 4 syllables/sec
        self.default_audio_token_syllable_ratio = self.frame_rate / 4
        self.max_audio_frames = 750
        # if no silence, how many extra words to inpaint on each side of actual change
        self.dynamic_word_buffer = 1
        # how many extra words to show the model on each side of the inpainted region
        self.static_word_buffer = 5
        # try to preserve silences longer than this (don't extend dynamic word buffer)
        self.break_spacing_time = 0.5

        self.max_tts_text_input_length = 500

        self.timer = Timer()

    def load_preset(
        self,
        hf_repo_id: str = "PlayHT/inpainter",
        vocoder_path: str = "v090_g_01105000",
        tokenizer_path: str = "tokenizer-multi_bpe16384_merged_extended_58M.json",
        speech_tokenizer_path: str = "xlsr2_1b_v2_custom.pt",
        kmeans_layer_path: str = "kmeans_10k.npy",
        voice_encoder_path: str = "voice_encoder_1992000.pt",
        inpainter_path: str = "last_250k_fixed.pkl",
    ) -> dict:
        from huggingface_hub import hf_hub_download

        vocoder_path = hf_hub_download(repo_id=hf_repo_id, filename=vocoder_path)
        tokenizer_path = hf_hub_download(repo_id=hf_repo_id, filename=tokenizer_path)
        speech_tokenizer_path = hf_hub_download(repo_id=hf_repo_id, filename=speech_tokenizer_path)
        kmeans_layer_path = hf_hub_download(repo_id=hf_repo_id, filename=kmeans_layer_path)
        voice_encoder_path = hf_hub_download(repo_id=hf_repo_id, filename=voice_encoder_path)
        inpainter_path = hf_hub_download(repo_id=hf_repo_id, filename=inpainter_path)

        preset = dict(
            vocoder = dict(
                checkpoint = vocoder_path,
                kmeans_layer_checkpoint = kmeans_layer_path,
            ),
            tokenizer = dict(
                vocab_file = tokenizer_path,
            ),
            speech_tokenizer = dict(
                checkpoint = speech_tokenizer_path,
                kmeans_layer_checkpoint = kmeans_layer_path,
                sample_rate = 16000,
            ),
            voice_encoder = dict(
                checkpoint = voice_encoder_path,
            ),
            inpainter = dict(
                checkpoint = inpainter_path,
            ),
        )

        return preset

    def handle_word_times(self, input_word_times, input_text: str):
        import jiwer

        input_text = ''.join([c for c in input_text if c.isalnum() or c.isspace() or c == "'" or c == "-"])
        print(f"Text for word times alignment: {input_text}")

        if not isinstance(input_word_times, list):
            raise ValueError("input_word_times must be a list")
        for word in input_word_times:
            if not isinstance(word, dict):
                raise ValueError("input_word_times must be a list of dictionaries")
            if word.get("word") is None or not isinstance(word["word"], str):
                raise ValueError(
                    "input_word_times must contain a 'word' key with string value"
                )
        word_times = [
            {
                "word": word["word"].lower(),
                "start": word.get("start"),
                "end": word.get("end"),
            }
            for word in input_word_times
        ]

        # if the aligner missed words, add them to the word_times list
        timed_words = [entry["word"].lower() for entry in word_times]
        word_times_align = jiwer.process_words(
            input_text.lower(), " ".join(timed_words)
        )
        last_hyp_index = 0
        word_times_mod = []
        for chunk in word_times_align.alignments[0]:
            if chunk.type != "equal":
                for i in range(last_hyp_index, chunk.hyp_start_idx):
                    word = word_times[i]
                    word_times_mod.append(word)
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    word = word_times_align.references[0][i]
                    word_times_mod.append({"word": word})
                last_hyp_index = chunk.hyp_end_idx
        for i in range(last_hyp_index, len(word_times)):
            word_times_mod.append(word_times[i])

        print("Word times (added missing words):")
        for i, word in enumerate(word_times_mod):
            print(f"    {i}: {word}")

        assert len(word_times_mod) == len(word_times_align.references[0])

        return word_times_mod

    def calculate_audio_token_syllable_ratio(self, word_times: List[Dict]):
        import syllables

        speech_time = 0.0
        last_word_end = -self.break_spacing_time
        words_with_times = []
        for i, word in enumerate(word_times):
            if (
                word["word"] == "<|unknown|>"
                or word.get("start") is None
                or word.get("end") is None
            ):
                continue
            words_with_times.append(word["word"])
            if word["start"] - last_word_end < self.break_spacing_time:
                speech_time += word["end"] - last_word_end
            else:
                speech_time += word["end"] - word["start"] + self.break_spacing_time
            last_word_end = word["end"]
        print(f"Speech time: {speech_time}")
        n_syllables = syllables.estimate(" ".join(words_with_times))
        print(f"Number of syllables: {n_syllables}")
        if n_syllables == 0:
            audio_token_syllable_ratio = self.default_audio_token_syllable_ratio
        else:
            audio_token_syllable_ratio = speech_time * self.frame_rate / n_syllables
        print(f"Audio token to syllable ratio: {audio_token_syllable_ratio}")
        return audio_token_syllable_ratio

    def calculate_diff_words(self, text_align, word_times: List[Dict], input_audio_tokens):
        merged_chunks = []  # type: ignore
        last_chunk_end = None
        for chunk in text_align.alignments[0]:
            print(chunk)
            if chunk.type != "equal":
                ref_start = chunk.ref_start_idx
                hyp_start = chunk.hyp_start_idx
                ref_end = chunk.ref_end_idx
                hyp_end = chunk.hyp_end_idx
                ref_diff_start = chunk.ref_start_idx
                ref_diff_end = chunk.ref_end_idx

                # add dynamic word buffer to the start/end of the ref/hyp
                # unless there is a long silence at some point
                for i in range(0, self.dynamic_word_buffer):
                    # break if we're at the start or if we hit a long silence
                    if ref_start - i <= 0:
                        break
                    # for an insertion at the very end, use the end time of the audio
                    if ref_start - i == len(word_times):
                        current_word_start = (
                            input_audio_tokens.shape[-1] * self.frame_rate
                        )
                    else:
                        current_word_start = word_times[ref_start - i].get("start")
                    prev_word_end = word_times[ref_start - i - 1].get("end")
                    if (
                        current_word_start is not None
                        and prev_word_end is not None
                        and current_word_start - prev_word_end > self.break_spacing_time
                    ):
                        break
                    ref_start -= 1
                    if hyp_start > 0:
                        hyp_start -= 1
                for i in range(0, self.dynamic_word_buffer):
                    # break if we're at the end or if we hit a long silence
                    if ref_end + i >= len(word_times):
                        break
                    current_word_end = word_times[ref_end + i - 1].get("end")
                    next_word_start = word_times[ref_end + i].get("start")
                    if (
                        current_word_end is not None
                        and next_word_start is not None
                        and next_word_start - current_word_end > self.break_spacing_time
                        # if it's an insert, assume it goes on the "right" of a silence
                        # TODO use punctuation to be smarter about this
                        and not (chunk.type == "insert" and i == 0)
                    ):
                        break
                    ref_end += 1
                    if hyp_end < len(text_align.hypotheses[0]):
                        hyp_end += 1

                # if start/end word is missing a time, step back/forward til we find one
                while ref_start > 0 and ref_start < len(word_times) and word_times[ref_start].get("start") is None:
                    ref_start -= 1
                    if hyp_start > 0:
                        hyp_start -= 1
                while (
                    ref_end < len(word_times)
                    and word_times[ref_end - 1].get("end") is None
                ):
                    ref_end += 1
                    if hyp_end < len(text_align.hypotheses[0]):
                        hyp_end += 1

                # add static word buffer to the start/end of the ref/hyp
                if ref_start - self.static_word_buffer < 0:
                    ref_buf_start = 0
                else:
                    ref_buf_start = ref_start - self.static_word_buffer
                if hyp_start - self.static_word_buffer < 0:
                    hyp_buf_start = 0
                else:
                    hyp_buf_start = hyp_start - self.static_word_buffer
                if ref_end + self.static_word_buffer > len(word_times):
                    ref_buf_end = len(word_times)
                else:
                    ref_buf_end = ref_end + self.static_word_buffer
                if hyp_end + self.static_word_buffer > len(text_align.hypotheses[0]):
                    hyp_buf_end = len(text_align.hypotheses[0])
                else:
                    hyp_buf_end = hyp_end + self.static_word_buffer

                # if start/end buf word is missing a time,
                # step back/forward til we find one
                while (
                    ref_buf_start > 0
                    and word_times[ref_buf_start].get("start") is None
                ):
                    ref_buf_start -= 1
                    if hyp_buf_start > 0:
                        hyp_buf_start -= 1
                while (
                    ref_buf_end < len(word_times)
                    and word_times[ref_buf_end - 1].get("end") is None
                ):
                    ref_buf_end += 1
                    if hyp_buf_end < len(text_align.hypotheses[0]):
                        hyp_buf_end += 1

                # if the chunk is adjacent to the previous chunk, merge them
                if last_chunk_end is not None and ref_start <= last_chunk_end:
                    merged_chunks[-1].ref_end = ref_end
                    merged_chunks[-1].ref_buf_end = ref_buf_end
                    merged_chunks[-1].hyp_end = hyp_end
                    merged_chunks[-1].hyp_buf_end = hyp_buf_end
                    merged_chunks[-1].ref_diff_end = ref_diff_end
                    print("Merged with previous chunk")
                    print(
                        f"Ref text: \
                            {text_align.references[0][merged_chunks[-1].ref_start:merged_chunks[-1].ref_end]}"
                    )
                    print(
                        f"Hyp text: \
                            {text_align.hypotheses[0][merged_chunks[-1].hyp_start:merged_chunks[-1].hyp_end]}"
                    )
                    print(
                        f"Ref text (buffered): \
                            {text_align.references[0][merged_chunks[-1].ref_buf_start:merged_chunks[-1].ref_buf_end]}"
                    )
                    print(
                        f"Hyp text (buffered): \
                            {text_align.hypotheses[0][merged_chunks[-1].hyp_buf_start:merged_chunks[-1].hyp_buf_end]}"
                    )
                else:
                    diff_chunk = TextDiffChunk(
                        ref_start=ref_start,
                        ref_end=ref_end,
                        hyp_start=hyp_start,
                        hyp_end=hyp_end,
                        ref_buf_start=ref_buf_start,
                        ref_buf_end=ref_buf_end,
                        hyp_buf_start=hyp_buf_start,
                        hyp_buf_end=hyp_buf_end,
                        ref_diff_start=ref_diff_start,
                        ref_diff_end=ref_diff_end,
                    )
                    print("Added new chunk")
                    print(
                        f"Ref text: \
                            {text_align.references[0][diff_chunk.ref_start:diff_chunk.ref_end]}"
                    )
                    print(
                        f"Hyp text: \
                            {text_align.hypotheses[0][diff_chunk.hyp_start:diff_chunk.hyp_end]}"
                    )
                    print(
                        f"Ref text (buffered): \
                            {text_align.references[0][diff_chunk.ref_buf_start:diff_chunk.ref_buf_end]}"
                    )
                    print(
                        f"Hyp text (buffered): \
                            {text_align.hypotheses[0][diff_chunk.hyp_buf_start:diff_chunk.hyp_buf_end]}"
                    )
                    merged_chunks.append(diff_chunk)
                last_chunk_end = ref_end

        return merged_chunks

    def calculate_diff_frames(
            self,
            merged_chunks: List[TextDiffChunk],
            text_align,
            word_times: List[Dict],
            input_audio_tokens,
            audio_token_syllable_ratio: float,
    ):
        import numpy as np
        import syllables

        diffs = []
        for chunk in merged_chunks:
            ref_start = chunk.ref_start
            hyp_start = chunk.hyp_start
            ref_end = chunk.ref_end
            hyp_end = chunk.hyp_end
            ref_buf_start = chunk.ref_buf_start
            hyp_buf_start = chunk.hyp_buf_start
            ref_buf_end = chunk.ref_buf_end
            hyp_buf_end = chunk.hyp_buf_end
            ref_diff_start = chunk.ref_diff_start
            ref_diff_end = chunk.ref_diff_end

            # compute the start/end frames of the inpainted region and the static buffer
            # account for long silences at the start/end of the file
            if ref_start == 0:
                # if we don't have a timestamp for the first word, we are forced to
                # assume it's at the start of the file
                if word_times[0].get("start") is None:
                    start_frame = 0
                    start_silence_frames = 0
                else:
                    start_frame = max(
                        0, int(word_times[0]["start"] * self.frame_rate)
                    )
                    start_silence_frames = min(
                        start_frame, int(self.break_spacing_time * self.frame_rate)
                    )
            # pure insertion at the end
            elif ref_start == len(word_times):
                start_frame = input_audio_tokens.shape[-1]
                start_silence_frames = 0
            else:
                start_frame = int(word_times[ref_start]["start"] * self.frame_rate)
                # if preceded by silence, inpaint up to half in case timestamp is off
                if (
                    word_times[ref_start - 1].get("end") is not None
                    and word_times[ref_start]["start"]
                    - word_times[ref_start - 1]["end"]
                    > self.break_spacing_time
                ):
                    silence_seconds = (
                        word_times[ref_start]["start"]
                        - word_times[ref_start - 1]["end"]
                    ) / 2
                    start_silence_frames = min(
                        int(silence_seconds * self.frame_rate),
                        int(self.break_spacing_time * self.frame_rate),
                    )
                else:
                    start_silence_frames = 0
            if ref_end == len(word_times) and ref_end != ref_start:
                # if we don't have a timestamp for the last word, we are forced to
                # assume it's at the end of the file
                if word_times[-1].get("end") is None:
                    end_frame = input_audio_tokens.shape[-1]
                    end_silence_frames = 0
                else:
                    end_frame = min(
                        input_audio_tokens.shape[-1],
                        int(word_times[-1]["end"] * self.frame_rate),
                    )
                    end_silence_frames = min(
                        input_audio_tokens.shape[-1] - end_frame,
                        int(self.break_spacing_time * self.frame_rate),
                    )
            else:
                # pure insertion
                if ref_end == ref_start:
                    end_frame = start_frame
                else:
                    end_frame = int(
                        word_times[ref_end - 1]["end"] * self.frame_rate
                    )
                # if followed by a silence, mask half of it in case the timestamp is off
                if (
                    ref_end < len(word_times)
                    and word_times[ref_end].get("start") is not None
                    and word_times[ref_end]["start"]
                    - word_times[ref_end - 1]["end"]
                    > self.break_spacing_time
                ):
                    silence_seconds = (
                        word_times[ref_end]["start"]
                        - word_times[ref_end - 1]["end"]
                    ) / 2
                    end_silence_frames = min(
                        int(silence_seconds * self.frame_rate),
                        int(self.break_spacing_time * self.frame_rate),
                    )
                else:
                    end_silence_frames = 0
            if ref_buf_start == 0:
                buf_start_frame = 0
            else:
                buf_start_frame = int(
                    word_times[ref_buf_start]["start"] * self.frame_rate
                )
            if ref_buf_end == len(word_times):
                buf_end_frame = input_audio_tokens.shape[-1]
            else:
                buf_end_frame = int(
                    word_times[ref_buf_end - 1]["end"] * self.frame_rate
                )

            # only inpaint silence if the actual diff is at the start/end of the
            # region to inpaint (i.e. no dynamic buffer)
            # we do this in case the timestamps are off, and stuff we want to get rid of
            # is left out of the region to inpaint
            # but if we have a dynamic buffer, it's fine, the inpainter will handle it
            if ref_diff_start == ref_start:
                start_silence_frames = min(
                    start_silence_frames, start_frame - buf_start_frame
                )
            else:
                start_silence_frames = 0
            if ref_diff_end == ref_end:
                end_silence_frames = min(end_silence_frames, buf_end_frame - end_frame)
            else:
                end_silence_frames = 0
            start_frame -= start_silence_frames
            end_frame += end_silence_frames

            # keep track of the text at each step, to tokenize and pass to the inpainter
            ref_text = text_align.references[0][ref_start:ref_end]
            hyp_text = text_align.hypotheses[0][hyp_start:hyp_end]
            print(f"Ref text: {ref_text}")
            print(f"Hyp text: {hyp_text}")
            ref_text_buf = text_align.references[0][ref_buf_start:ref_buf_end]
            hyp_text_buf = text_align.hypotheses[0][hyp_buf_start:hyp_buf_end]
            print(f"Ref text (buffered): {ref_text_buf}")
            print(f"Hyp text (buffered): {hyp_text_buf}")
            text_to_submit = (
                ref_text_buf[: ref_start - ref_buf_start]
                + hyp_text
                + ref_text_buf[ref_end - ref_buf_start :]
            )
            print(f"Text to submit: {text_to_submit}")

            # determine number of phonemes and estimate number of frames for the output
            n_syllables = syllables.estimate(" ".join(hyp_text))
            print(f"Hyp syllables: {n_syllables}")
            n_frames = int(n_syllables * audio_token_syllable_ratio)
            print(f"N frames: {n_frames}")

            # crudely split the diff into multiple chunks if it's too long
            if n_frames > self.max_audio_frames:
                n_subdiffs = 1 + n_frames // self.max_audio_frames
                sub_words = np.array_split(text_to_submit, n_subdiffs)
                for i in range(n_subdiffs):
                    if i == 0:
                        hyp_text = " ".join(sub_words[i][self.static_word_buffer:])
                        sub_start_frame = start_frame
                        sub_end_frame = None
                        sub_buf_start_frame = buf_start_frame
                        sub_buf_end_frame = None
                    elif i == n_subdiffs - 1:
                        hyp_text = " ".join(sub_words[i][:-self.static_word_buffer])
                        sub_start_frame = None
                        sub_end_frame = end_frame
                        sub_buf_start_frame = None
                        sub_buf_end_frame = buf_end_frame
                    else:
                        hyp_text = " ".join(sub_words[i])
                        sub_start_frame = None
                        sub_end_frame = None
                        sub_buf_start_frame = None
                        sub_buf_end_frame = None
                    n_syllables = syllables.estimate(hyp_text)
                    n_frames = int(n_syllables * audio_token_syllable_ratio)
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
                    )
                )
        print(f"{len(diffs)} diffs to inpaint")
        return diffs

    def do_inpaint(self, diffs: List[InpainterChunk], input_audio_tokens, input):
        import torch

        with torch.inference_mode():
            output_chunks = []
            last_end_frame = -1
            for chunk in diffs:
                start_frame = chunk.start_frame
                end_frame = chunk.end_frame
                n_frames = chunk.n_frames
                buf_start_frame = chunk.buf_start_frame
                buf_end_frame = chunk.buf_end_frame
                text_tokens = chunk.text_tokens
                start_time = (
                    start_frame / self.frame_rate if start_frame is not None else -1
                )
                end_time = end_frame / self.frame_rate if end_frame is not None else -1
                print(
                    f"Generating inpainted audio for {start_frame} ({start_time:.2f}s) \
                    to {end_frame} ({end_time:.2f}s) with {n_frames} inpainted frames \
                    ({n_frames / self.frame_rate:.2f}s)"
                )
                print(f"Text tokens shape: {text_tokens.shape}")
                text = self.mm.tokenizer.decode_tokens_tensor(text_tokens)
                print(f"Text: {text}")

                # save the previous unchanged audio tokens
                if start_frame is not None and start_frame > last_end_frame + 1:
                    output_chunks.append(
                        input_audio_tokens[:, last_end_frame + 1 : start_frame]
                    )
                    print(f"Unchanged audio tokens shape: {output_chunks[-1].shape}")
                if end_frame is not None:
                    last_end_frame = end_frame

                # assemble the tokens to pass to the inpainter
                # possibly including static buffer at begin and/or end
                tokens_to_cat = []
                if buf_start_frame is not None and start_frame is not None:
                    tokens_to_cat.append(
                        input_audio_tokens[:, buf_start_frame:start_frame]
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
                    tokens_to_cat.append(input_audio_tokens[:, end_frame:buf_end_frame])
                    inpainter_end_frame = inpainter_start_frame + n_frames
                else:
                    inpainter_end_frame = -1
                inpaint_audio_tokens = torch.cat(tokens_to_cat, dim=1)
                if inpainter_end_frame == -1:
                    inpainter_end_frame = inpaint_audio_tokens.shape[1]

                # inpaint the desired region, with the static buffer unmodified
                print(
                    f"Pre-inpaint audio tokens shape: {inpaint_audio_tokens.shape} \
                        ({inpaint_audio_tokens.shape[1] / self.frame_rate:.2f}s)"
                )
                print(
                    f"PlayDiffusion start frame: {inpainter_start_frame} \
                        ({inpainter_start_frame / self.frame_rate:.2f}s)"
                )
                print(
                    f"PlayDiffusion end frame: {inpainter_end_frame} \
                        ({inpainter_end_frame / self.frame_rate:.2f}s)"
                )
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
                    inpainter_start_frame : inpainter_start_frame + n_frames,
                ]
                print(
                    f"Trimmed inpainted audio token shape: {inpaint_audio_tokens.shape}"
                )
                output_chunks.append(inpaint_audio_tokens)

        # save the remaining unchanged audio tokens and concatenate all chunks
        if input_audio_tokens.shape[-1] > last_end_frame + 1:
            output_chunks.append(input_audio_tokens[:, last_end_frame + 1 :])
            print(f"Unchanged audio tokens shape: {output_chunks[-1].shape}")
        return torch.cat(output_chunks, dim=1)


    def inpaint(self, input: InpaintInput):
        import jiwer
        import torch
        import torchaudio.functional as F
        from unidecode import unidecode

        self.timer.reset()

        print(f"Input: {input}")

        # normalize the input text
        input_text = unidecode(input.input_text)
        output_text = unidecode(input.output_text)
        print(f"Inpainter input text: {input_text}")
        print(f"Inpainter output text: {output_text}")
        self.timer("Normalize text")

        # determine differences between input and output text
        text_align = jiwer.process_words(input_text, output_text)
        self.timer("Align text")

        # use the input audio itself for the vocoder embedding
        vocoder_emb = get_vocoder_embedding(input.audio, self.mm).to(self.device)
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

        # calculate alignment to determine word times
        word_times = self.handle_word_times(input.input_word_times, input_text)
        self.timer("Handle word times")

        # determine audio-token-to-syllable ratio (excluding long silences)
        if input.audio_token_syllable_ratio is not None:
            audio_token_syllable_ratio = input.audio_token_syllable_ratio
        else:
            audio_token_syllable_ratio = self.calculate_audio_token_syllable_ratio(word_times)
        self.timer("Audio token to syllable ratio")

        # merge adjacent diff chunks and calculate static buffer to pass to inpainter
        merged_chunks = self.calculate_diff_words(text_align, word_times, input_audio_tokens)
        self.timer("Prepare word times and diff chunks")

        # iterate over the diffs, calculating inputs for the inpainter
        diffs = self.calculate_diff_frames(merged_chunks, text_align, word_times, input_audio_tokens, audio_token_syllable_ratio)
        self.timer("Calculate diffs to inpaint")

        # generate inpainted audio
        output_audio_tokens = self.do_inpaint(diffs, input_audio_tokens, input)
        self.timer("Inpainter")

        # vocode the output audio
        with torch.inference_mode():
            audio_g = self.mm.vocoder(output_audio_tokens, vocoder_emb)
        self.timer("Vocoder")

        # encode audio for output
        return (self.mm.vocoder.output_frequency, make_16bit_pcm(audio_g).squeeze())

    def do_split(self, text: str, delimiters: Optional[str] = None):
        if delimiters is None:
            return self.split_text_as_necessary(text[:len(text)//2]) + self.split_text_as_necessary(text[len(text)//2:])

        midpoint = len(text)//2
        for i in range(midpoint):
            high = midpoint + i
            if high < len(text) - 1 and text[high] in delimiters:
                return self.split_text_as_necessary(text[:high+1]) + self.split_text_as_necessary(text[high+1:])
            low = midpoint - i - 1
            if low > 0 and text[low] in delimiters:
                return self.split_text_as_necessary(text[:low+1]) + self.split_text_as_necessary(text[low+1:])

        # failed to split on these delimiters
        return None

    def split_text_as_necessary(self, text: str):
        text = text.strip()
        if len(text) <= self.max_tts_text_input_length:
            return [text]

        on_sentence_boundary = self.do_split(text, '.!?')
        if on_sentence_boundary is not None:
            return on_sentence_boundary

        on_midsentence_pause = self.do_split(text, ',;:')
        if on_midsentence_pause is not None:
            return on_midsentence_pause

        on_space = self.do_split(text, ' ')
        if on_space is not None:
            return on_space

        return self.do_split(text)

    def tts(self, input: TTSInput):
        import syllables
        import torch
        from unidecode import unidecode

        self.timer.reset()

        print(f"Input: {input}")

        # normalize the input text
        output_text = unidecode(input.output_text)
        print(f"TTS text: {output_text}")
        split_texts = self.split_text_as_necessary(output_text)
        print(f"Split texts:")
        for text in split_texts:
            print(f"    {text}")
        self.timer("Normalize and split text")

        with torch.inference_mode():
            vocoder_emb = get_vocoder_embedding(input.voice, self.mm).to(self.device)
            self.timer("Get vocoder embedding")

            tts_result_tokens = []
            for text in split_texts:
                # tokenize the text
                text_tokens = self.mm.tokenizer.encode_normalized_to_tensor(text)
                self.timer("Tokenize")

                # estimate the number of frames for the TTS result
                ratio = input.audio_token_syllable_ratio or self.default_audio_token_syllable_ratio
                n_syllables = syllables.estimate(text)
                target_len = int(n_syllables * ratio)
                self.timer("Estimate frames")

                # generate the TTS result
                print(f"Generating TTS with {target_len} frames")
                tts_result_tokens.append(self.mm.inpainter.generate(
                    text_tokens=text_tokens,
                    target_len=target_len,
                    n_timesteps=input.num_steps,
                    init_temp=input.init_temp,
                    init_diversity=input.init_diversity,
                    guidance=input.guidance,
                    rescale_cfg=input.rescale,
                    topk=input.topk,
                ))
                self.timer("TTS generation")

            audio_g = self.mm.vocoder(torch.cat(tts_result_tokens, dim=1), vocoder_emb)
            self.timer("Vocoder")

        return (self.mm.vocoder.output_frequency, make_16bit_pcm(audio_g).squeeze())

    def rvc(self, input: RVCInput):
        import torch
        import torchaudio.functional as F

        self.timer.reset()

        print(f"Input: {input}")
        
        # get target voice's vocoder_emb
        vocoder_emb = get_vocoder_embedding(input.target_voice, self.mm).to(self.device)
        self.timer("Get vocoder embedding")

        # extract xlsr audio tokens
        input_wav, sr = load_audio(input.source_speech, self.device)
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

        # vocode the output audio
        with torch.inference_mode():
            audio_g = self.mm.vocoder(input_audio_tokens, vocoder_emb)
        self.timer("Vocoder")

        return (self.mm.vocoder.output_frequency, make_16bit_pcm(audio_g).squeeze())