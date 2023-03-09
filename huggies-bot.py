import streamlit as st
import pandas as pd
import numpy as np
import openai
from PIL import Image

image = Image.open('fotor_2023-3-9_15_18_29.png')
st.image(image, width = 200)
st.title("Baby Bot")

st.markdown("""---""")

openai.api_key = st.secrets["OPENAI_API_KEY"]

embeddings = pd.read_csv("embeddings.csv")
embeddings = embeddings.drop(embeddings.columns[0], axis=1)

def get_embedding(text):
    result = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return result["data"][0]["embedding"]
def compute_doc_embeddings(df):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.query) for idx, r in df.iterrows()
    }
def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))
def calc_sim(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    sim_score = []
    for i in range(len(contexts.iloc[0:])):
        sim_score.append((contexts.iloc[i][1],vector_similarity(query_embedding,contexts.iloc[i][2:])))
    sim_score.sort(key=lambda x: x[1], reverse=True)
    return sim_score


#models
def davinciC(query):    
    #query = How to feed my baby in the first year
    ss = calc_sim(query, embeddings)
    context = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
    prompt =f"""Answer the question in as many words and as truthfully as possible using the provided context

    Context:
    {context}

    Q: {query}
    A:"""

    base_model = "text-davinci-003"
    completion = openai.Completion.create(
        model = base_model,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion.choices[0].text)
def davinciNC(query):     
    base_model = "text-davinci-003"
    completion = openai.Completion.create(
        model = base_model,
        prompt = query,
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion.choices[0].text)
def turbo(query):
    #query = "How to feed my baby in the first year"
    ss = calc_sim(query, embeddings)
    context = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
    base_model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        model = base_model,
        messages=[
            {"role": "system", "content": "You are a chatbot that will provide answers with the help of the assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context}
        ],
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion['choices'][0]['message']['content'])



#UI
st.markdown(
    """
<style>
.css-fblp2m {
    fill: rgb(255 255 255);
}
.css-18ni7ap {
    background: #0f059e;
}
.css-1avcm0n {
    background: #0f059e;
}
</style>
""",
    unsafe_allow_html=True,
)
#image = Image.open('logo.png')
#st.image(image, width=400)

st.sidebar.info('Please choose the model from the dropdown below.')
st.set_option('deprecation.showfileUploaderEncoding', False)
#add_selectbox = st.sidebar.selectbox("Which model would you like to use?", ("gpt-3.5-turbo", "text-davinci-003", "no context - davinci"))
add_selectbox = st.sidebar.selectbox("", ("Customized GPT3", "Default GPT3","Customized ChatGPT (Experimental)"))
st.sidebar.write('Note: Some models have been trained with select public content from www.huggies.com')


st.write('On the day you bring your newborn baby home, life as you know it changes forever. We have put all tips, techniques and information in one place, to help make newborn baby care as easy as possible for new parents')
if add_selectbox == "Customized ChatGPT (Experimental)":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask The Bot"):
        output = turbo(text1)
        st.success(output)
elif add_selectbox == "Customized GPT3":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask The Bot"):
        output = davinciC(text1)
        st.success(output)
elif add_selectbox == "Default GPT3":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask The Bot"):
        output = davinciNC(text1)
        st.success(output)
