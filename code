# Streamlit Version of Chatbot for Multi-File + Image + Video Support Using Gemini

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import os
import io
from PIL import Image
import tempfile
import moviepy.editor as mp
import speech_recognition as sr
import subprocess

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------
# Utility Functions
# ------------------------

# Standardized prompt template for consistent formatting across all modalities
STANDARD_PROMPT_TEMPLATE = """
Provide a to-the-point answer using the provided information.
- ALWAYS use bullet points or numbered lists when possible
- Keep each point concise and specific
- Avoid long sentences - break into clear points
- Include all necessary information in point format
- If answer not available: 'Information not available.'

{content_type}:
{content}

Question:
{question}

Answer:
"""


def extract_text_from_file(uploaded_file):
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        content = uploaded_file.read()
        text = ""
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    # Add spacing between pages
                    text += extracted + "\n\n"

        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                if para.text.strip():  # Only add non-empty paragraphs
                    text += para.text + "\n"

        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
            text = df.to_string(index=False)

        else:
            raise ValueError(f"Unsupported file type: {filename}")

        if not text.strip():
            # Better logging for debugging
            print(f"WARNING: No text extracted from {uploaded_file.name}")
            st.warning(
                f"⚠️ No text could be extracted from {uploaded_file.name}")
            return ""

        # Log successful extraction
        print(
            f"SUCCESS: Extracted {len(text)} characters from {uploaded_file.name}")
        return text

    except Exception as e:
        # Better error logging
        print(f"ERROR processing {uploaded_file.name}: {str(e)}")
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return ""


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Save human-readable copy for debugging
    debug_file_path = "processed_documents_debug.txt"
    with open(debug_file_path, "w", encoding="utf-8") as debug_file:
        debug_file.write("=" * 50 + "\n")
        debug_file.write("PROCESSED DOCUMENTS DEBUG FILE\n")
        debug_file.write(f"Generated on: {pd.Timestamp.now()}\n")
        debug_file.write(f"Total chunks: {len(text_chunks)}\n")
        debug_file.write("=" * 50 + "\n\n")

        for i, chunk in enumerate(text_chunks, 1):
            debug_file.write(f"CHUNK {i}:\n")
            debug_file.write("-" * 30 + "\n")
            debug_file.write(chunk)
            debug_file.write("\n" + "-" * 30 + "\n\n")

    if os.path.exists("faiss_index"):
        db = FAISS.load_local(
            "faiss_index", embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_texts(text_chunks)
    else:
        db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local("faiss_index")

    print(f"Debug file saved: {debug_file_path}")


def get_conversational_chain():
    # Using standardized template for document context
    template = STANDARD_PROMPT_TEMPLATE.format(
        content_type="Context",
        content="{context}",
        question="{question}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    prompt = PromptTemplate(
        template=template, input_variables=[
            "context", "question"
        ])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def ask_from_documents(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        return "Please process documents first."

    db = FAISS.load_local(
        "faiss_index", embeddings,
        allow_dangerous_deserialization=True
    )

    # Add debugging information
    docs = db.similarity_search(question, k=8)
    print(f"DEBUG: Found {len(docs)} relevant chunks for question: {question}")

    # Check if any documents were found
    if not docs:
        return "No relevant information found in the documents."

    chain = get_conversational_chain()
    result = chain({
        "input_documents": docs, "question": question
    }, return_only_outputs=True)

    # Add response validation
    output = result.get("output_text", "").strip()
    if not output:
        return "Unable to generate a response from the available context."

    return output


def ask_from_image(image_file, question):
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        # Using standardized prompt template for consistency
        prompt = STANDARD_PROMPT_TEMPLATE.format(
            content_type="Image Analysis",
            content="Analyze the provided image",
            question=question
        )
        response = model.generate_content([prompt, image])

        if response.text:
            return response.text
        else:
            return "• Information not available."

    except Exception as e:
        return f"• Error processing image: {str(e)}"


def convert_to_wav(input_path, output_path):
    try:
        result = subprocess.run([
            "ffmpeg", "-i", input_path,
            "-ac", "1", "-ar", "16000", "-f", "wav", output_path, "-y"
        ], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg conversion failed: {e.stderr}")
        return False
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install FFmpeg to process videos.")
        return False


def ask_from_video(video_file, question):
    temp_files = []  # Track temporary files for cleanup
    try:
        # Reset file pointer to beginning
        video_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
            temp_files.append(video_path)

        clip = mp.VideoFileClip(video_path)
        if clip.audio is None:
            return "❌ No audio track found in the video."

        raw_audio_path = video_path.replace(".mp4", ".wav")
        temp_files.append(raw_audio_path)
        clip.audio.write_audiofile(raw_audio_path, verbose=False, logger=None)
        clip.close()  # Free up resources

        converted_audio_path = video_path.replace(".mp4", "_converted.wav")
        temp_files.append(converted_audio_path)

        if not convert_to_wav(raw_audio_path, converted_audio_path):
            return "❌ Failed to convert audio format."

        recognizer = sr.Recognizer()
        with sr.AudioFile(converted_audio_path) as source:
            audio = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio)
        except sr.RequestError as e:
            return f"❌ API request failed: {e}"
        except sr.UnknownValueError:
            return "❌ Could not understand audio."

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        # Using standardized prompt template for consistency
        prompt = STANDARD_PROMPT_TEMPLATE.format(
            content_type="Video Transcript",
            content=f"Transcript: {transcript}",
            question=question
        )
        response = model.generate_content(prompt)

        # Add proper response validation
        if response.text:
            return response.text
        else:
            return "• Information not available."

    except Exception as e:
        return f"❌ Error processing video: {str(e)}"
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


# --------------------------
# Sidebar: Upload + Process
# --------------------------
with st.sidebar:
    st.header("📎 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / Excel files", type=[
            "pdf", "docx", "xls", "xlsx"
        ], accept_multiple_files=True)

    if st.button("📄 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_text = ""
                processed_files = []
                failed_files = []

                for f in uploaded_files:
                    # Reset file pointer before processing
                    f.seek(0)
                    text = extract_text_from_file(f)
                    if text:  # Only process if text was successfully extracted
                        # Add proper spacing between documents
                        all_text += text + "\n\n=== DOCUMENT SEPARATOR ===\n\n"
                        processed_files.append(f.name)
                    else:
                        failed_files.append(f.name)

                if all_text.strip():
                    chunks = get_text_chunks(all_text)
                    get_vector_store(chunks)
                    st.success(
                        f"✅ Successfully processed {len(processed_files)} documents!")
                    if processed_files:
                        st.info(
                            f"📄 Processed files: {', '.join(processed_files)}")
                    if failed_files:
                        st.warning(
                            f"⚠️ Failed to process: {', '.join(failed_files)}")
                else:
                    st.error(
                        "❌ No text could be extracted from the uploaded files.")
        else:
            st.warning("⚠️ No files uploaded.")

# --------------------------
# Text QA
# --------------------------
st.subheader("💬 Ask from Documents")
user_question = st.text_input("Ask your question here", key="doc_question")
if st.button("Ask", key="doc_ask"):
    if user_question.strip():
        with st.spinner("Searching documents..."):
            answer = ask_from_documents(user_question)
            st.write("📘", answer)
    else:
        st.warning("⚠️ Please enter a question.")

# --------------------------
# Image QA
# --------------------------
st.subheader("🖼️ Ask from Image")
img_file = st.file_uploader("Upload an image", type=[
                            "png", "jpg", "jpeg"], key="image_uploader")
img_question = st.text_input("Question about the image", key="img_question")

if st.button("Process Image", key="img_process"):
    if img_file and img_question.strip():
        with st.spinner("Analyzing image..."):
            answer = ask_from_image(img_file, img_question)
            st.image(img_file, caption="Uploaded Image", use_column_width=True)
            st.write("🧠", answer)
    elif not img_file:
        st.warning("⚠️ Please upload an image.")
    else:
        st.warning("⚠️ Please enter a question about the image.")

# --------------------------
# Video QA
# --------------------------
st.subheader("🎥 Ask from Video")
video_file = st.file_uploader("Upload a video", type=[
                              "mp4", "mov"], key="video_uploader")
video_question = st.text_input(
    "Question about the video", key="video_question")

if st.button("Process Video", key="video_process"):
    if video_file and video_question.strip():
        with st.spinner("Extracting audio and generating transcript..."):
            answer = ask_from_video(video_file, video_question)
            st.video(video_file)
            st.write("🧠", answer)
    elif not video_file:
        st.warning("⚠️ Please upload a video.")
    else:
        st.warning("⚠️ Please enter a question about the video.")
