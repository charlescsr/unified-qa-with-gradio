import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
import gradio as gr
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "allenai/unifiedqa-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(question, passage, **generator_args):
    input_string = str(question) + " \\n " + str(passage)
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)[0]

gr.Interface(run_model, inputs=["text", "textbox"], outputs=["text"]).launch()