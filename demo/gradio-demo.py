import os

import gradio as gr
from openai import OpenAI

from playdiffusion import PlayDiffusion, InpaintInput, TTSInput, RVCInput
import whisper_timestamped as whisper

inpainter = PlayDiffusion()

def get_whisper_client(backend_choice,api_key):
    if backend_choice == "OpenAI Whisper API":
        if not api_key or not api_key.strip():
            raise gr.Error("OpenAI API key is required and cannot be empty.")
        else:
            _whisper_client = OpenAI(api_key=api_key.strip())
    else:
        _whisper_client = whisper

    return _whisper_client

def run_asr(audio, backend_choice="Local Whisper", local_model="tiny",api_key=""):
    if audio is None:
        raise gr.Error("Please upload or record an audio file before running ASR.")
    whisper_client = get_whisper_client(backend_choice,api_key)

    if backend_choice == "OpenAI Whisper API":
        try:
            with open(audio, "rb") as audio_file:
                transcript = whisper_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            word_times = [{
                "word": word.word,
                "start": word.start,
                "end": word.end
            } for word in transcript.words]
            return transcript.text, transcript.text, word_times
        except Exception as e:
            raise gr.Error(f"Failed to call OpenAI Whisper API: {e}")
    else:
        audio_data = whisper_client.load_audio(audio)
        model = whisper_client.load_model(local_model)
        transcript = whisper_client.transcribe(model, audio_data, language="en")
        transcript_text = transcript.get("text", "")
        word_times = []
        for segment in transcript.get("segments", []):
            for word in segment.get("words", []):
                word_times.append({
                    "word": word["text"],
                    "start": word["start"],
                    "end": word["end"]
                })
        return transcript_text, transcript_text, word_times

def run_inpainter(input_text, output_text, word_times, audio, num_steps, init_temp, init_diversity, guidance, rescale, topk, use_manual_ratio, audio_token_syllable_ratio):
    if not use_manual_ratio:
        audio_token_syllable_ratio = None
    return inpainter.inpaint(InpaintInput(input_text=input_text, output_text=output_text, input_word_times=word_times, audio=audio, num_steps=num_steps,
                                          init_temp=init_temp, init_diversity=init_diversity, guidance=guidance, rescale=rescale, topk=topk,
                                          audio_token_syllable_ratio=audio_token_syllable_ratio))

def run_inpainter_tts(input_text, voice_audio, num_steps, init_temp, init_diversity, guidance, rescale, topk, use_manual_ratio, audio_token_syllable_ratio):
    if not use_manual_ratio:
        audio_token_syllable_ratio = None
    return inpainter.tts(TTSInput(output_text=input_text, voice=voice_audio, num_steps=num_steps, init_temp=init_temp,
                                  init_diversity=init_diversity, guidance=guidance, rescale=rescale, topk=topk,
                                  audio_token_syllable_ratio=audio_token_syllable_ratio))

def toggle_ratio_input(use_manual):
    return gr.update(visible=use_manual, interactive=use_manual)

def create_advanced_options_accordion():
    with gr.Accordion("Advanced options", open=False):
        num_steps_slider = gr.Slider(1, 100, 30, step=1, label="number of sampling steps codebook")
        init_temp_slider = gr.Slider(0.5, 10, 1, step=0.1, label="Initial temperature")
        init_diversity_slider = gr.Slider(0, 10, 1, step=0.1, label="Initial diversity")
        guidance_slider = gr.Slider(0, 10, 0.5, step=0.1, label="guidance")
        rescale_slider = gr.Slider(0, 1, 0.7, step=0.1, label="guidance rescale factor")
        topk_slider = gr.Slider(1, 10000, 25, step=1, label="sampling from top-k logits")

        gr.Markdown("#### Audio Token Syllable Ratio")
        gr.Markdown("*Automatic calculation (recommended) provides the best results in most cases.*")
        use_manual_ratio = gr.Checkbox(label="Use manual audio token syllable ratio", value=False)
        audio_token_syllable_ratio = gr.Number(
            label="Audio token syllable ratio (manual)",
            value=12.5, precision=2, minimum=5.0, maximum=25.0,
            visible=False, interactive=False
        )
        use_manual_ratio.change(
            toggle_ratio_input,
            inputs=[use_manual_ratio],
            outputs=[audio_token_syllable_ratio]
        )

    return (num_steps_slider, init_temp_slider, init_diversity_slider,
            guidance_slider, rescale_slider, topk_slider,
            use_manual_ratio, audio_token_syllable_ratio)


def speech_rvc(rvc_source_speech, rvc_target_voice):
    return inpainter.rvc(RVCInput(source_speech=rvc_source_speech, target_voice=rvc_target_voice))



if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False, title="PlayDiffusion") as demo:
        gr.Markdown("## PlayDiffusion")

        with gr.Tab("Inpaint"):
            gr.Markdown("### Upload an audio file and run ASR to get the text.")
            gr.Markdown("### Then, specify the desired output text.")
            gr.Markdown("### Run the inpainter to generate the modified audio.")
            gr.Markdown("### Note: The model and demo are currently targeted for English.")

            inpaint_advanced_options = create_advanced_options_accordion()

            with gr.Row():
                asr_backend = gr.Radio(
                    ["Local Whisper", "OpenAI Whisper API"], value="Local Whisper", label="ASR Backend"
                )

                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key here",
                    visible=False
                )

                whisper_model = gr.Dropdown(
                    ["tiny", "base", "small", "medium", "large", "turbo"],
                    value="tiny",
                    label="Local Whisper Model",
                    interactive=True
                )

            with gr.Row():
                audio_input = gr.Audio(label="Upload audio to be modified", sources=["upload", "microphone"], type="filepath")

            with gr.Row():
                asr_submit = gr.Button("Run ASR")

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Input text from ASR", interactive=False)
                    text_output = gr.Textbox(label="Desired output text")
                with gr.Column():
                    word_times = gr.JSON(label="Word times from ASR")

            with gr.Row():
                inpainter_submit = gr.Button("Run Inpainter")

            with gr.Row():
                audio_output = gr.Audio(label="Output audio")

            # Show/hide OpenAI key or local model selection
            def toggle_asr_inputs(backend_choice):
                return {
                    openai_api_key: gr.update(visible=(backend_choice == "OpenAI Whisper API"),interactive=(backend_choice == "OpenAI Whisper API")),
                    whisper_model: gr.update(visible=(backend_choice == "Local Whisper"),interactive=(backend_choice == "Local Whisper")),
                }

            asr_submit.click(
                run_asr,
                inputs=[audio_input, asr_backend, whisper_model,openai_api_key],
                outputs=[text_input, text_output, word_times]
            )
            inpainter_submit.click(
                run_inpainter,
                inputs=[text_input, text_output, word_times, audio_input] + list(inpaint_advanced_options),
                outputs=[audio_output])

            asr_backend.change(
                toggle_asr_inputs,
                inputs=[asr_backend],
                outputs=[openai_api_key,whisper_model]
            )

        with gr.Tab("Text to Speech"):
            gr.Markdown("### Text to Speech")
            tts_advanced_options = create_advanced_options_accordion()

            tts_text = gr.Textbox(label="TTS Input", placeholder="Enter text to convert to speech", lines=2)
            tts_voice =  gr.Audio(label="Voice to use for TTS",
                sources=["upload", "microphone"], type="filepath",
            )
            tts_submit = gr.Button("Convert to Speech")
            tts_output = gr.Audio(label="Generated Speech")

            tts_submit.click(
                run_inpainter_tts,
                inputs=[tts_text, tts_voice] + list(tts_advanced_options),
                outputs=[tts_output]
            )

        with gr.Tab("Voice Conversion"):
            gr.Markdown("### Real Time Voice Conversion (works best for english)")
            rvc_source_speech =  gr.Audio(label="Source Conversion Speech",
                sources=["upload", "microphone"], type="filepath",
            )
            rvc_target_voice =  gr.Audio(label="Target Voice",
                sources=["upload", "microphone"], type="filepath",
            )
            rvc_submit = gr.Button("Real time Voice Conversion")
            rvc_output = gr.Audio(label="Converted Speech")

            rvc_submit.click(
                speech_rvc,
                inputs=[rvc_source_speech, rvc_target_voice],
                outputs=[rvc_output]
            )

    demo.launch(share=True)
