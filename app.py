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

gr.Interface(run_model, inputs=["text", "textbox"], 
        outputs=["text"], 
        examples=[["When did Tesla first show off the turbine?", 
        "On his 50th birthday in 1906, Tesla demonstrated his 200 horsepower (150 kilowatts) 16,000 rpm bladeless turbine. During 1910–1911 at the Waterside Power Station in New York, several of his bladeless turbine engines were tested at 100–5,000 hp."], 
        ["What type of musical instruments did the Yuan bring to China?", "Western musical instruments were introduced to enrich Chinese performing arts. From this period dates the conversion to Islam, by Muslims of Central Asia, of growing numbers of Chinese in the northwest and southwest. Nestorianism and Roman Catholicism also enjoyed a period of toleration. Buddhism (especially Tibetan Buddhism) flourished, although Taoism endured certain persecutions in favor of Buddhism from the Yuan government. Confucian governmental practices and examinations based on the Classics, which had fallen into disuse in north China during the period of disunity, were reinstated by the Yuan court, probably in the hope of maintaining order over Han society. Advances were realized in the fields of travel literature, cartography, geography, and scientific education."]]).launch()