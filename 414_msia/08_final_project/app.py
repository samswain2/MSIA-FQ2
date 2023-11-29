import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import time
from langchain.document_loaders import WebBaseLoader


# Load environment variables
load_dotenv()
chatGPT_api_key = os.getenv("CHATGPT_API_KEY")

st.set_page_config(page_title='Content Summarizer', layout='wide')

# Use consistent colors and layout design
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Content Summarizer', anchor=None)
st.image("picture.png", width=500, caption="Created by ChatGPT: Innovating Content Summarization")
st.markdown('<p class="big-font">Enter Content Details</p>', unsafe_allow_html=True)

# Use a sidebar for user inputs
with st.sidebar:
    st.markdown('## Content Options')
    choice = st.radio("Select Content Type:", ('YouTube Video', 'Web Page'))
    url = st.text_input("URL:", help='Paste the URL of the content you want to summarize.')


if choice == 'YouTube Video' and url:


    with st.spinner('Fetching video information...'):
        youtube_loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        video_docs = youtube_loader.load()
        video_metadata = video_docs[0].metadata

   
    st.subheader("Video Information")
    video_title = video_metadata['title']
    st.markdown(f"**Title:** {video_title}")

    video_count = video_metadata['view_count']
    st.markdown(f"**Views:** {video_count:,}")

    video_length = video_metadata['length']
    minutes, seconds = divmod(video_length, 60)
    st.markdown(f"**Length:** {minutes} minutes and {seconds} seconds")

    video_author = video_metadata['author']
    st.markdown(f"**Author:** {video_author}")

    video_publish_date = video_metadata['publish_date']
    st.markdown(f"**Publish Date:** {video_publish_date}")
    st.markdown(f"**Video Link:** {url}")

    # Define prompt for summary
    video_prompt_template = """
    Write a comprehensive summary of the following content from a YouTube video:
    "{text}"

    In your summary, please include:
    1. The main topic or theme of the video.
    2. Key details and important points discussed.
    3. Any significant context or background information that is relevant.
    4. Conclusions or calls to action, if any, presented in the video.

    SUMMARY:
    """
    video_prompt = PromptTemplate.from_template(video_prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", openai_api_key=chatGPT_api_key)
    video_llm_chain = LLMChain(llm=llm, prompt=video_prompt)

    # Define StuffDocumentsChain
    video_stuff_chain = StuffDocumentsChain(llm_chain=video_llm_chain, document_variable_name="text")


    # Generate and display summary
    with st.spinner('Generating summary, please wait...'):
        video_summary_information = video_stuff_chain.run(video_docs)

    with st.expander("Video Summary"):
        st.write(video_summary_information)
elif choice == 'Web Page' and url:

    # Define prompt
    web_prompt_template = """
    Write a concise and clear summary of the following text. Ensure that the summary:

    1. Captures the main points and key facts.
    2. Maintains the original tone and intent of the text.
    3. Is brief and avoids unnecessary details or repetition.
    4. Is well-structured for easy understanding (use bullet points or a numbered list if appropriate).

    Text to Summarize:
    "{text}"

    CONCISE SUMMARY:
    """

    web_prompt = PromptTemplate.from_template(web_prompt_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", openai_api_key=chatGPT_api_key)
    web_llm_chain = LLMChain(llm=llm, prompt=web_prompt)
    web_stuff_chain = StuffDocumentsChain(llm_chain=web_llm_chain, document_variable_name="text")


    try:
        with st.spinner('Fetching Website information...'):
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            web_docs_meta = web_docs[0].metadata

        st.subheader("Website Information")
        web_title = web_docs_meta['title']
        st.markdown(f"**Title:** {web_title}")

        web_language = web_docs_meta['language']
        st.markdown(f"**Language:** {web_language}")

        web_source = web_docs_meta['source']
        st.markdown(f"**Source:** {web_source}")
        
        # Generate the summary
        with st.spinner('Generating summary, please wait...'):
            web_summary_information = web_stuff_chain.run(web_docs)
        
        with st.expander("Website Summary"):
            st.write(web_summary_information)

    except Exception as e:
        st.write("Sorry, we're unable to process the content from the provided URL due to its size. Please try another URL with shorter content.")
        st.write(e)
else:
    st.warning('Please select a content type and enter a URL to generate a summary.')



