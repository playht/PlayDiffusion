from dataclasses import dataclass
from typing import List, Optional

from play_inpainter.models.model_manager import InpainterModelManager
from play_inpainter.pydantic_models.models import InpainterInput
from play_inpainter.utils.audio_utils import Timer, get_vocoder_embedding, load_audio
from play_inpainter.utils.save_audio import make_16bit_pcm


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


class Inpainter():
    def __init__(self, device: str = "cuda"):
        import nltk
        import torch

        self.device = torch.device(device)

        self.preset = self.load_preset()
        self.mm = InpainterModelManager(self.preset, self.device)

        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        self.frame_rate = 50
        # avg en speaker speaks 4 syllables/sec
        self.default_audio_token_syllable_ratio = self.frame_rate / 4
        self.max_audio_frames = 750

        self.timer = Timer()

    def load_preset(
        self,
        vocoder_path: str = "s3://play-fal/ldm-models/v090_g_01105000",
        tokenizer_path: str = "s3://play-fal/v3-models/tokenizer-multi_bpe16384_merged_extended_58M.json",
        speech_tokenizer_path: str = "s3://play-fal/ldm-models/xlsr2_1b_v2_custom.pt",
        kmeans_layer_path: str = "s3://play-fal/ldm-models/kmeans_10k.npy",
        voice_encoder_path: str = "s3://play-fal/ldm-models/voice_encoder_1992000.pt",
        inpainter_path: str = "s3://play-fal/ldm-models/last_250k_fixed.pkl",
    ) -> dict:
        from play_inpainter.utils.preset_util import get_all_checkpoints_from_preset

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

        preset = get_all_checkpoints_from_preset(preset)

        return preset

    def inpaint(self, input: InpainterInput):
        import jiwer
        import numpy as np
        import syllables
        import torch
        import torchaudio.functional as F

        self.timer.reset()

        print(f"Inpainter input text: {input.input_text}")
        print(f"Inpainter output text: {input.output_text}")
        print("Inpainter input word times:")
        for i, word in enumerate(input.input_word_times):
            print(f"    {i}: {word}")

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

        # validate word times
        if not isinstance(input.input_word_times, list):
            raise ValueError("input_word_times must be a list")
        for word in input.input_word_times:
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
            for word in input.input_word_times
        ]

        # determine audio-token-to-syllable ratio (excluding long silences)
        speech_time = 0.0
        break_spacing_time = 0.5
        last_word_end = -break_spacing_time
        words_with_times = []
        for i, word in enumerate(word_times):
            if (
                word["word"] == "<|unknown|>"
                or word["start"] is None
                or word["end"] is None
            ):
                continue
            words_with_times.append(word["word"])
            if word["start"] - last_word_end < break_spacing_time:
                speech_time += word["end"] - last_word_end
            else:
                speech_time += word["end"] - word["start"] + break_spacing_time
            last_word_end = word["end"]
        print(f"Speech time: {speech_time}")
        n_syllables = syllables.estimate(" ".join(words_with_times))
        print(f"Number of syllables: {n_syllables}")
        if n_syllables == 0:
            audio_token_syllable_ratio = self.default_audio_token_syllable_ratio
        else:
            audio_token_syllable_ratio = speech_time * self.frame_rate / n_syllables
        print(f"Audio token to syllable ratio: {audio_token_syllable_ratio}")
        self.timer("Audio token to syllable ratio")

        # if any word times are missing, add them to the word_times list
        timed_words = [entry["word"].lower() for entry in word_times]
        word_times_align = jiwer.process_words(
            input.input_text.lower(), " ".join(timed_words)
        )
        text_align = jiwer.process_words(input.input_text, input.output_text)
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
        assert len(word_times_mod) == len(word_times_align.references[0])
        print("Word times (added missing words):")
        for i, word in enumerate(word_times_mod):
            print(f"    {i}: {word}")

        # merge adjacent diff chunks and calculate static buffer to pass to inpainter
        merged_chunks = []  # type: ignore
        # if no silence, how many extra words to inpaint on each side of actual change
        dynamic_word_buffer = 1
        # how many extra words to show the model on each side of the inpainted region
        static_word_buffer = 5
        last_chunk_end = None
        for chunk in text_align.alignments[0]:
            if chunk.type != "equal":
                ref_start = chunk.ref_start_idx
                hyp_start = chunk.hyp_start_idx
                ref_end = chunk.ref_end_idx
                hyp_end = chunk.hyp_end_idx
                ref_diff_start = chunk.ref_start_idx
                ref_diff_end = chunk.ref_end_idx

                # add dynamic word buffer to the start/end of the ref/hyp
                # unless there is a long silence at some point
                for i in range(0, dynamic_word_buffer):
                    # break if we're at the start or if we hit a long silence
                    if ref_start - i <= 0:
                        break
                    # for an insertion at the very end, use the end time of the audio
                    if ref_start - i == len(word_times_mod):
                        current_word_start = (
                            input_audio_tokens.shape[-1] * self.frame_rate
                        )
                    else:
                        current_word_start = word_times_mod[ref_start - i].get("start")
                    prev_word_end = word_times_mod[ref_start - i - 1].get("end")
                    if (
                        current_word_start is not None
                        and prev_word_end is not None
                        and current_word_start - prev_word_end > break_spacing_time
                    ):
                        break
                    ref_start -= 1
                    if hyp_start > 0:
                        hyp_start -= 1
                for i in range(0, dynamic_word_buffer):
                    # break if we're at the end or if we hit a long silence
                    if ref_end + i >= len(word_times_mod):
                        break
                    current_word_end = word_times_mod[ref_end + i - 1].get("end")
                    next_word_start = word_times_mod[ref_end + i].get("start")
                    if (
                        current_word_end is not None
                        and next_word_start is not None
                        and next_word_start - current_word_end > break_spacing_time
                        # if it's an insert, assume it goes on the "right" of a silence
                        # TODO use punctuation to be smarter about this
                        and not (chunk.type == "insert" and i == 0)
                    ):
                        break
                    ref_end += 1
                    if hyp_end < len(text_align.hypotheses[0]):
                        hyp_end += 1

                # if start/end word is missing a time, step back/forward til we find one
                while ref_start > 0 and word_times_mod[ref_start].get("start") is None:
                    ref_start -= 1
                    if hyp_start > 0:
                        hyp_start -= 1
                while (
                    ref_end < len(word_times_mod)
                    and word_times_mod[ref_end - 1].get("end") is None
                ):
                    ref_end += 1
                    if hyp_end < len(text_align.hypotheses[0]):
                        hyp_end += 1

                # add static word buffer to the start/end of the ref/hyp
                if ref_start - static_word_buffer < 0:
                    ref_buf_start = 0
                else:
                    ref_buf_start = ref_start - static_word_buffer
                if hyp_start - static_word_buffer < 0:
                    hyp_buf_start = 0
                else:
                    hyp_buf_start = hyp_start - static_word_buffer
                if ref_end + static_word_buffer > len(word_times_mod):
                    ref_buf_end = len(word_times_mod)
                else:
                    ref_buf_end = ref_end + static_word_buffer
                if hyp_end + static_word_buffer > len(text_align.hypotheses[0]):
                    hyp_buf_end = len(text_align.hypotheses[0])
                else:
                    hyp_buf_end = hyp_end + static_word_buffer

                # if start/end buf word is missing a time,
                # step back/forward til we find one
                while (
                    ref_buf_start > 0
                    and word_times_mod[ref_buf_start].get("start") is None
                ):
                    ref_buf_start -= 1
                    if hyp_buf_start > 0:
                        hyp_buf_start -= 1
                while (
                    ref_buf_end < len(word_times_mod)
                    and word_times_mod[ref_buf_end - 1].get("end") is None
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
        self.timer("Prepare word times and diff chunks")

        # iterate over the diffs, calculating inputs for the inpainter
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
                if word_times_mod[0].get("start") is None:
                    start_frame = 0
                    start_silence_frames = 0
                else:
                    start_frame = max(
                        0, int(word_times_mod[0]["start"] * self.frame_rate)
                    )
                    start_silence_frames = min(
                        start_frame, int(break_spacing_time * self.frame_rate)
                    )
            else:
                start_frame = int(word_times_mod[ref_start]["start"] * self.frame_rate)
                # if preceded by silence, inpaint up to half in case timestamp is off
                if (
                    word_times_mod[ref_start - 1].get("end") is not None
                    and word_times_mod[ref_start]["start"]
                    - word_times_mod[ref_start - 1].get("end")
                    > break_spacing_time
                ):
                    silence_seconds = (
                        word_times_mod[ref_start]["start"]
                        - word_times_mod[ref_start - 1]["end"]
                    ) / 2
                    start_silence_frames = min(
                        int(silence_seconds * self.frame_rate),
                        int(break_spacing_time * self.frame_rate),
                    )
                else:
                    start_silence_frames = 0
            if ref_end == len(word_times_mod):
                # if we don't have a timestamp for the last word, we are forced to
                # assume it's at the end of the file
                if word_times_mod[-1].get("end") is None:
                    end_frame = input_audio_tokens.shape[-1]
                    end_silence_frames = 0
                else:
                    end_frame = min(
                        input_audio_tokens.shape[-1],
                        int(word_times_mod[-1]["end"] * self.frame_rate),
                    )
                    end_silence_frames = min(
                        input_audio_tokens.shape[-1] - end_frame,
                        int(break_spacing_time * self.frame_rate),
                    )
            else:
                # pure insertion
                if ref_end == ref_start:
                    end_frame = start_frame
                else:
                    end_frame = int(
                        word_times_mod[ref_end - 1]["end"] * self.frame_rate
                    )
                # if followed by a silence, mask half of it in case the timestamp is off
                if (
                    word_times_mod[ref_end].get("start") is not None
                    and word_times_mod[ref_end].get("start")
                    - word_times_mod[ref_end - 1]["end"]
                    > break_spacing_time
                ):
                    silence_seconds = (
                        word_times_mod[ref_end]["start"]
                        - word_times_mod[ref_end - 1]["end"]
                    ) / 2
                    end_silence_frames = min(
                        int(silence_seconds * self.frame_rate),
                        int(break_spacing_time * self.frame_rate),
                    )
                else:
                    end_silence_frames = 0
            if ref_buf_start == 0:
                buf_start_frame = 0
            else:
                buf_start_frame = int(
                    word_times_mod[ref_buf_start]["start"] * self.frame_rate
                )
            if ref_buf_end == len(word_times_mod):
                buf_end_frame = input_audio_tokens.shape[-1]
            else:
                buf_end_frame = int(
                    word_times_mod[ref_buf_end - 1]["end"] * self.frame_rate
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
                        hyp_text = " ".join(sub_words[i][static_word_buffer:])
                        sub_start_frame = start_frame
                        sub_end_frame = None
                        sub_buf_start_frame = buf_start_frame
                        sub_buf_end_frame = None
                    elif i == n_subdiffs - 1:
                        hyp_text = " ".join(sub_words[i][:-static_word_buffer])
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
                    f"Inpainter start frame: {inpainter_start_frame} \
                        ({inpainter_start_frame / self.frame_rate:.2f}s)"
                )
                print(
                    f"Inpainter end frame: {inpainter_end_frame} \
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
        self.timer("Inpainter")

        # save the remaining unchanged audio tokens and concatenate all chunks
        if input_audio_tokens.shape[-1] > last_end_frame + 1:
            output_chunks.append(input_audio_tokens[:, last_end_frame + 1 :])
            print(f"Unchanged audio tokens shape: {output_chunks[-1].shape}")
        output_audio_tokens = torch.cat(output_chunks, dim=1)

        # vocode the output audio
        with torch.no_grad():
            audio_g = self.mm.vocoder(output_audio_tokens, vocoder_emb)
        self.timer("Vocoder")

        # encode audio for output
        return (self.mm.vocoder.output_frequency, make_16bit_pcm(audio_g).squeeze())
