import os
import re

import gradio as gr
from openai import OpenAI

from playdiffusion import PlayDiffusion, InpaintInput, TTSInput
from playdiffusion.utils.audio_utils import raw_audio_to_torch_audio
from playdiffusion.utils.save_audio import make_16bit_pcm
from playdiffusion.utils.voice_resource import VoiceResource

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

def run_inpainter(input_text, output_text, word_times, audio, num_steps, init_temp, init_diversity, guidance, rescale, topk):
    return inpainter.inpaint(InpaintInput(input_text=input_text, output_text=output_text, input_word_times=word_times, audio=audio, num_steps=num_steps, init_temp=init_temp, init_diversity=init_diversity, guidance=guidance, rescale=rescale, topk=topk))

def run_inpainter_tts(input_text, voice_audio):
    return inpainter.tts(TTSInput(output_text=input_text, voice=voice_audio))

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False, title="PlayDiffusion") as demo:
        gr.Markdown("## PlayDiffusion")

        with gr.Tab("Inpaint"):
            gr.Markdown("### Upload an audio file and run ASR to get the text.")
            gr.Markdown("### Then, specify the desired output text.")
            gr.Markdown("### Run the inpainter to generate the modified audio.")
            gr.Markdown("### Note: The model and demo are currently targeted for English.")

            with gr.Accordion("Advanced options", open=False):
                num_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="number of sampling steps codebook", value=30)
                init_temp_slider = gr.Slider(minimum=0.5, maximum=10, step=0.1, label="Initial temperature", value=1)
                init_diversity_slider = gr.Slider(minimum=0, maximum=10, step=0.1, label="Initial diversity", value=1)
                guidance_slider = gr.Slider(minimum=0, maximum=10, step=0.1, label="guidance", value=0.5)
                rescale_slider = gr.Slider(minimum=0, maximum=1, step=0.1, label="guidance rescale factor", value=0.7)
                topk_slider = gr.Slider(minimum=1, maximum=10000, step=1, label="sampling from top-k logits", value=25)

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
            inpainter_submit.click(run_inpainter, inputs=[text_input, text_output, word_times, audio_input, num_steps_slider, init_temp_slider, init_diversity_slider, guidance_slider, rescale_slider, topk_slider], outputs=[audio_output])

        with gr.Tab("Text to Speech"):
            gr.Markdown("### Text to Speech")
            tts_text = gr.Textbox(label="TTS Input", placeholder="Enter text to convert to speech", lines=2)
            tts_voice =  gr.Audio(label="Voice to use for TTS",
                sources=["upload", "microphone"], type="filepath",
            )
            tts_submit = gr.Button("Convert to Speech")
            tts_output = gr.Audio(label="Generated Speech")

            tts_submit.click(
                run_inpainter_tts,
                inputs=[tts_text, tts_voice],
                outputs=[tts_output]
            )

    demo.launch(share=True)
