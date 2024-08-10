import streamlit as st
from src.utils import get_youtube_transcript, create_vector_db, setup_rag_chain

st.header("Youtube Video Query")

if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:    
    video_url = st.text_input("Enter Youtube video URL")
    video_id = video_url.split("v=")[1] if "v=" in video_url else ""

    if st.button("Process Video") and video_id:
        st.session_state.video_id = video_id
        with st.spinner('Processing...'):
            # Get Transcript
            transcript = get_youtube_transcript(video_id)
            
            # create vector database
            st.session_state.vector_store = create_vector_db(transcript)
            st.success("Done")

if st.session_state.vector_store:
    # set up RAG chain
    rag_chain = setup_rag_chain(st.session_state.vector_store)
    
    query = st.text_input("Enter your query")
    if query:
        # Get answer
        result = rag_chain({"query": query})
        st.write(result['result'])
