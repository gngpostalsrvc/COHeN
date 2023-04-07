import re
import gradio as gr
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline
)
from transformers_interpret import SequenceClassificationExplainer
from hebrewtools.functions import sbl_normalization

model_name = 'gngpostalsrvc/COHeN'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
cls_explainer = SequenceClassificationExplainer(model, tokenizer)

pipe = pipeline("text-classification", model=model_name)

pattern = 
re.compile("[^\s\u05d0-\u05ea\u05b0-\u05bc\u05be\u05c1\u05c2\u05c7]")

def predict(text):
  text = " ".join([word for word in text.split() if word not in ['\u05e1', 
'\u05e4', '']])
  text = re.sub(pattern, "", text)
  text = sbl_normalization(text)
  word_attributions = cls_explainer(text)
  results = pipe(text)[0]
  label = f"{results['label']} ({results['score']:.2})"
  return label, word_attributions[1:-1]

iface = gr.Interface(
    fn=predict,
    inputs=gr.Text(label="Input Text"),
    outputs=[gr.Text(label="Label"), gr.HighlightedText(label="Word 
Importance", show_legend=True).style(color_map={"-": "red", "+": 
"green"})],
    theme=gr.themes.Base(),
    examples=['וּבִשְׁנַת אַחַת לְכוֺרֶשׁ מֶלֶךְ פָּרַס לִכְלוֺת דְּבַר־יְהוָה מִפִּי יִרְמְיָה הֵעִיר יְהוָה 
אֶת־רוּחַ כֹּרֶשׁ מֶלֶךְ־פָּרַס וַיַּעֲבֶר־קוֺל בְּכָל־מַלְכוּתוֺ וְגַם־בְּמִכְתָּב לֵאמֹר', 'וַיֹּאמֶר דָּוִד 
אֶל־אוּרִיָּה שֵׁב בָּזֶה גַּם־הַיּוֺם וּמָחָר אֲשַׁלְּחֶךָּ וַיֵּשֶׁב אוּרִיָּה בִירוּשָׁלִַם בַּיּוֺם הַהוּא וּמִמָּחֳרָת']
)

iface.launch()
