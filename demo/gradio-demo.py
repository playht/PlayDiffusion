import os
from io import BytesIO

import gradio as gr
from openai import OpenAI

from play_inpainter import Inpainter, InpainterInput
from play_inpainter.utils.save_audio import make_16bit_pcm

whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
inpainter = Inpainter()

def run_asr(audio):
    #_, audio_bytes = audio
    #audio_file = BytesIO(make_16bit_pcm(audio_bytes))
    #audio_file.name = "audio.wav"
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
    return inpainter.inpaint(InpainterInput(input_text=input_text, output_text=output_text, input_word_times=word_times, audio=audio, num_steps=num_steps, init_temp=init_temp, init_diversity=init_diversity, guidance=guidance, rescale=rescale, topk=topk))

if __name__ == '__main__':

    with gr.Blocks(analytics_enabled=False, title="Play Inpainter") as demo:
        gr.Markdown("## Play Inpainter")
        gr.Markdown("### Upload an audio file and run ASR to get the text.")
        gr.Markdown("### Then, specify the desired output text.")
        gr.Markdown("### Run the inpainter to generate the modified audio.")

        with gr.Accordion("Advanced options", open=False):
            num_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="number of sampling steps codebook", value=30)
            init_temp_slider = gr.Slider(minimum=0.5, maximum=10, step=0.1, label="Initial temperature", value=1)
            init_diversity_slider = gr.Slider(minimum=0, maximum=10, step=0.1, label="Initial diversity", value=1)
            guidance_slider = gr.Slider(minimum=0, maximum=10, step=0.1, label="guidance", value=0.5)
            rescale_slider = gr.Slider(minimum=0, maximum=1, step=0.1, label="guidance rescale factor", value=0.7)
            topk_slider = gr.Slider(minimum=1, maximum=10000, step=1, label="sampling from top-k logits", value=25)

        with gr.Row():
            audio_input = gr.Audio(label="Upload audio to be modified", sources=["upload"], type="filepath")

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

    # Launch the Gradio app
    demo.launch(share=True)
