import os

import gradio as gr
from openai import OpenAI

from playdiffusion import PlayDiffusion, InpaintInput, TTSInput, RVCInput

whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
inpainter = PlayDiffusion()

def run_asr(audio):
    audio_file = open(audio, "rb")
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

            asr_submit.click(run_asr, inputs=[audio_input], outputs=[text_input, text_output, word_times])
            inpainter_submit.click(
                run_inpainter,
                inputs=[text_input, text_output, word_times, audio_input] + list(inpaint_advanced_options),
                outputs=[audio_output])

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
