# End-to-End-Youtube-RAG-GenAI-Application
End to end youtube RAG application using all the free tools: Langchain, Gemini API, GoogleGenAIEmbeddings, FAISS, Streamlit 

# YouTubeRAG: AI-Powered Video Content Exploration

YouTubeRAG is an intelligent application that leverages AI to provide insightful answers to questions about YouTube video content. By combining the power of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), this tool offers a unique way to interact with and extract information from YouTube videos.

## Features

- **YouTube Transcript Extraction**: Automatically fetches and processes transcripts from YouTube videos.
- **AI-Powered Question Answering**: Utilizes Google's Gemini LLM to generate accurate responses to user queries.
- **Retrieval-Augmented Generation**: Enhances AI responses by grounding them in the specific content of the video transcript.
- **Efficient Information Retrieval**: Uses FAISS for fast and efficient similarity search in the vector database.

## How It Works

1. The user provides a YouTube video URL.
2. The application extracts the video's transcript.
3. The transcript is processed and stored in a FAISS vector database.
4. Users can ask questions about the video content.
5. The application retrieves relevant information from the transcript and generates an AI-powered response.

## Technologies Used

- **Langchain**: For building the RAG pipeline.
- **Google Gemini**: Serves as the Large Language Model for question answering.
- **YouTube Transcript API**: For extracting video transcripts.
- **FAISS**: For efficient similarity search and information retrieval.
- **GoogleGenAIEmbeddings**: For creating text embeddings.

## How to use  
first run 'pip install -r requirements.txt' to install all the dependencies then get GoogleAPI key and store it in .env file then from terminal type: ' streamlit run app.py '

