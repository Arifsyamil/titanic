import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import base64
import uuid

import transformers
#from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
#Page header, title
st.set_page_config(page_title="Named Entity Recognition Tagger", page_icon="📘")
st.title("📘 Named Entity Recognition Tagger")


#Load model
#Use "roberta-large" based on article
#Previous use model "bert-base-uncased"
st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForTokenClassification.from_pretrained("roberta-large")
    trainer = Trainer(model=model)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    return trainer, model, tokenizer


#Tag generation
id2tag={0: 'O',
        1: 'B-corporation',
        2: 'I-corporation',
        3: 'B-creative-work',
        4: 'I-creative-work',
        5: 'B-group',
        6: 'I-group',
        7: 'B-location',
        8: 'I-location',
        9: 'B-person',
        10: 'I-person',
        11: 'B-product',
        12: 'I-product'}

def tag_sentence(text:str):
      # convert our text to a tokenized sequence
      inputs = tokenizer(text, truncation=True, return_tensors="pt")
      # get outputs
      outputs = model(**inputs)
      # convert to probabilities with softmax
      probs = outputs[0][0].softmax(1)
      # get the tags with the highest probability
      word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()], np.round(probs[i][tagid].item() *100,2) ) 
                    for i, tagid in enumerate (probs.argmax(axis=1))]

      df=pd.DataFrame(word_tags, columns=['word', 'tag', 'probability'])
      return df

#Download button
def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')

def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    return json_string


#Create form
with st.form(key='my_form'):

    x1 = st.text_input(label='Enter a sentence:', max_chars=250)
    submit_button = st.form_submit_button(label='🏷️ Create tags')

if submit_button:
    if re.sub('\s+','',x1)=='':
        st.error('Please enter a non-empty sentence.')

    elif re.match(r'\A\s*\w+\s*\Z', x1):
        st.error("Please enter a sentence with at least one word")
    
    else:
        st.markdown("### Tagged Sentence")
        st.header("")

        Trainer, model, tokenizer = load_model()  
        results=tag_sentence(x1)
        
        cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])

        with c1:
            #csvbutton = download_button(results, "results.csv", "📥 Download .csv")
            csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(results), file_name= "results.csv", mime='text/csv', key='csv')
        with c2:
            #textbutton = download_button(results, "results.txt", "📥 Download .txt")
            textbutton = st.download_button(label="📥 Download .txt", data=convert_df(results), file_name= "results.text", mime='text/plain',  key='text')
        with c3:
            #jsonbutton = download_button(results, "results.json", "📥 Download .json")
            jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(results), file_name= "results.json", mime='application/json',  key='json')

        st.header("")
        
        c1, c2, c3 = st.columns([1, 3, 1])
        
        with c2:

             st.table(results.style.background_gradient(subset=['probability']).format(precision=2))

#Apple announces the new MacBook Air, supercharged by the new ARM-based M2 chip
#Empire State building is located in New York, a city in United States
#About model
with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   The **Named Entity Recognition Tagger** app is a tool that performs named entity recognition.
-   The available entitites are: *corporation*, *creative-work*, *group*, *location*, *person* and *product*.
-   The app uses the [RoBERTa model](https://huggingface.co/roberta-large), fine-tuned on the [wnut](https://huggingface.co/datasets/wnut_17) dataset.      
-   The model uses the **byte-level BPE tokenizer**. Each sentece is first tokenized.
-   For more info regarding the data science part, check this [post](https://towardsdatascience.com/named-entity-recognition-with-deep-learning-bert-the-essential-guide-274c6965e2d?sk=c3c3699e329e45a8ed93d286ae04ef10).      
       """
    )