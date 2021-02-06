import streamlit as st
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

def text_analyzer(my_text):
    nlp= spacy.load('en_core_web_sm')
    docx= nlp(my_text)
    tokens=[token.text for token in docx]
    all_data= [('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
    return all_data



def entity_extr(my_text):
    nlp= spacy.load('en_core_web_sm')
    docx= nlp(my_text)
    entities=[(entity.text,entity.label_) for entity in docx.ents]
    return entities 



def main():
    st.title("NLPiffy With Streamlit")
    st.subheader(" Implementing Natural language processing")

#tokenization
    if st.checkbox("Show tokens and lemma"):
        st.subheader("Tokenize your text")
        message2=st.text_area("Enter text")
        if st.button("Analyze"):
            nlp_result= text_analyzer(message2)
            st.json(nlp_result)
    

# Named entity extraction

    if st.checkbox("Named entity extraction"):
            st.subheader("Extract your entities")
            message=st.text_area("Enter your text")
            if st.button("Extracting"):
                nlp_result= entity_extr(message)
                st.json(nlp_result)

# summarizattion

    if st.checkbox("Summarize your data"):
            st.subheader("show summary")
            message3=st.text_area("your text",)
            if st.button("Summarizing"):
                nlp_result= summarize(message3)
                st.success(nlp_result)


    if st.checkbox("Show sentiment"):
        
        st.subheader("sentiment Analysis")
        message1=st.text_area("Enter your text","Type here")
        if st.button("Analyzing"):
            blob= TextBlob(message1)
            result=blob.sentiment
            st.success(result)
    
if __name__== '__main__':
    main()