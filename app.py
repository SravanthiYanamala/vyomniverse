import streamlit as st
import google.generativeai as genai
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import fitz
import hashlib
import tempfile
from functools import lru_cache
from docx import Document  # Add this with your other imports

# Load environment variables
dotenv.load_dotenv()

# Gemini Models
google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
]

@lru_cache(maxsize=1)
def get_api_key():
    """Get API key with caching for deployment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            show_api_key_warning()
    return api_key

def get_file_hash(file_bytes):
    """Generate a hash for file content"""
    return hashlib.md5(file_bytes).hexdigest()

def requires_pro_model(messages):
    """Check if the conversation requires Pro model capabilities"""
    for message in messages:
        for content in message["content"]:
            # Always use Pro for videos and PDF OCR
            if content["type"] == "video_file":
                return True
            # Use Pro for complex image analysis requests
            if content["type"] == "image_url" and any(
                msg["content"][0]["text"].lower() in ["analyze", "describe", "explain"] 
                for msg in messages if msg["content"][0]["type"] == "text"
            ):
                return True
    return False

def messages_to_gemini(messages):
    """Convert Streamlit messages format to Gemini format"""
    gemini_messages = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        parts = []
        
        for content in message["content"]:
            if content["type"] == "text":
                parts.append({"text": content["text"]})
            elif content["type"] == "image_url":
                parts.append({"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": content["image_url"]["url"].split(",")[1]
                }})
            elif content["type"] == "video_file":
                with open(content["video_file"], "rb") as f:
                    video_data = f.read()
                parts.append({
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": base64.b64encode(video_data).decode("utf-8")
                    }
                })
            elif content["type"] == "audio_file":
                with open(content["audio_file"], "rb") as f:
                    audio_data = f.read()
                parts.append({"inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(audio_data).decode("utf-8")
                }})
        
        gemini_messages.append({
            "role": role,
            "parts": parts
        })
    return gemini_messages

def stream_llm_response(model_params, api_key):
    """Stream response from Gemini API with automatic model selection"""
    try:
        # Check for special identity queries first
        last_user_message = st.session_state.messages[-1] if st.session_state.messages else None
        
        if last_user_message and last_user_message["role"] == "user":
            user_content = last_user_message["content"][0]["text"].lower() if last_user_message["content"][0]["type"] == "text" else ""
            
            # Only respond with identity if it's a direct "who are you" question 
            # and not about uploaded content
            if ("who are you" in user_content or "what is this" in user_content) and not any(
                content["type"] in ["image_url", "video_file", "audio_file"] 
                for message in st.session_state.messages 
                for content in message["content"]
            ):
                response = (
                    "I'm VyomniVerse, your AI assistant powered by Google's Gemini technology. "
                    "VyomniVerse is an innovative project designed to provide intelligent conversations "
                    "with advanced multimedia capabilities. I can analyze images, videos, PDFs, and more!"
                )
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": [{
                        "type": "text",
                        "text": response,
                    }]
                })
                yield response
                return

        genai.configure(api_key=api_key)
        
        # Determine which model to use
        use_pro_model = requires_pro_model(st.session_state.messages)
        model_name = "gemini-1.5-pro" if use_pro_model else model_params["model"]
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": model_params.get("temperature", 0.3),
            }
        )
        
        gemini_messages = messages_to_gemini(st.session_state.messages)
        
        # Enhanced prompts for complex media analysis
        if use_pro_model:
            if any("inline_data" in part and part["inline_data"]["mime_type"].startswith("video/") 
                  for message in gemini_messages for part in message["parts"]):
                gemini_messages[-1]["parts"].insert(0, {
                    "text": "Analyze this video thoroughly and provide:\n"
                            "1. Scene-by-scene description\n"
                            "2. Spoken dialogue and captions\n"
                            "3. Important visual text\n"
                            "4. Overall context and meaning"
                })
            elif any("inline_data" in part and part["inline_data"]["mime_type"].startswith("image/")
                    for message in gemini_messages for part in message["parts"]):
                gemini_messages[-1]["parts"].insert(0, {
                    "text": "Analyze this image in detail including:\n"
                            "1. Key objects and their relationships\n"
                            "2. Text content if present\n"
                            "3. Overall context and meaning"
                })
        
        response = model.generate_content(
            contents=gemini_messages,
            stream=True
        )
        
        response_message = ""
        for chunk in response:
            if chunk.text:
                response_message += chunk.text
                yield chunk.text

        st.session_state.messages.append({
            "role": "assistant", 
            "content": [{
                "type": "text",
                "text": response_message,
            }]
        })
        
    except Exception as e:
        yield f"⚠️ Error: {str(e)}. Please check your API key and try again."


def add_image_to_messages():
    """Handle camera image capture and add to conversation"""
    if "camera_img" in st.session_state and st.session_state.camera_img:
        file_bytes = st.session_state.camera_img.getvalue()
        file_hash = get_file_hash(file_bytes)
        
        if file_hash not in st.session_state.processed_files:
            st.session_state.processed_files.add(file_hash)
            
            # Save the image
            image = Image.open(BytesIO(file_bytes))
            img_str = get_image_base64(image)
            
            # Add to messages
            st.session_state.messages.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                }]
            })
            st.rerun()

            
def get_image_base64(image_raw):
    """Convert image to base64 string"""
    buffered = BytesIO()
    if image_raw.mode == 'RGBA':
        image_raw = image_raw.convert('RGB')
    image_raw.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_from_pdf(file, api_key):
    """Extract text from PDF using both digital text and OCR"""
    text = ""
    try:
        # First try extracting digital text
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # If no text found, try OCR
        if not text.strip():
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            
            file.seek(0)
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base64.b64encode(base_image["image"]).decode("utf-8")
                    
                    result = model.generate_content([
                        "Extract text from this image:", 
                        {"mime_type": "image/jpeg", "data": image_bytes}
                    ])
                    if result.text:
                        text += result.text + "\n"
                        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX files"""
    try:
        doc = Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
        return ""


def extract_text_from_url(url):
    """Extract text content from a webpage"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

def download_chat_transcript():
    """Generate downloadable chat transcript"""
    transcript = ""
    for message in st.session_state.messages:
        role = message["role"]
        for content in message["content"]:
            if content["type"] == "text":
                transcript += f"{role}: {content['text']}\n\n"
    return transcript

def show_api_key_warning():
    """Display instructions for setting API key"""
    st.warning("""
        ⚠️ API Key Not Found  
        To use this app, please set your Google API Key:
        
        1. **Local Development**:  
           Create a `.env` file with:  
           `GOOGLE_API_KEY=your_key_here`
        
        2. **Cloud Deployment**:  
           Add to your secrets as `GOOGLE_API_KEY`
    """)
    st.stop()


def handle_media_upload(uploaded_file):
    """Handle media upload with memory optimization"""
    try:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type.split("/")[0]
        
        if file_type == "image":
            with Image.open(BytesIO(file_bytes)) as image:
                img_str = get_image_base64(image)
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                }
        elif file_type == "video":
            # For deployment, consider limiting video size
            if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
                st.error("Video too large (max 50MB)")
                return None
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(file_bytes)
                st.session_state.temp_files[tmp_file.name] = True
                return {
                    "type": "video_file",
                    "video_file": tmp_file.name
                }
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None



def main():
    st.set_page_config(
        page_title="The VyomniVerse",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <h1 style="text-align: center; color: #6ca395;">🤖 <i>The VyomniVerse</i> 💬</h1>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = {}

    # Get API key from secure source
    google_api_key = get_api_key()
    if not google_api_key:
        show_api_key_warning()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":      
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "audio_file":
                    st.audio(content["audio_file"])

    # Sidebar controls
    with st.sidebar:
        model = st.selectbox("Select a model:", google_models, index=0)
        
        with st.expander("⚙️ Model Parameters"):
            model_temp = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.1,
                help="Controls randomness. Lower = more deterministic"
            )

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        if st.button("🗑️ Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.processed_files = set()
            # Clean up temp files
            for file_path in st.session_state.temp_files.values():
                try:
                    os.unlink(file_path)
                except:
                    pass
            st.session_state.temp_files = {}
            st.rerun()

        st.divider()
        st.subheader("📤 Upload Content")
        
        # Unified file uploader for all media types
        uploaded_file = st.file_uploader(
            "Upload Files (Images, Videos, PDFs, DOCX)", 
            type=["png", "jpg", "jpeg", "mp4", "pdf", "docx"],
            key="unified_file_uploader",
            help="Upload images (PNG/JPG), videos (MP4), PDFs or Word documents (DOCX)"
        )

        # Handle the unified file upload
        if uploaded_file:
            file_bytes = uploaded_file.read()
            file_hash = get_file_hash(file_bytes)
            
            if file_hash not in st.session_state.processed_files:
                st.session_state.processed_files.add(file_hash)
                uploaded_file.seek(0)  # Reset file pointer after reading
                
                # Determine file type and process accordingly
                file_type = uploaded_file.type
                
                if file_type.startswith("image/"):
                    # Process as image
                    media_content = handle_media_upload(uploaded_file)
                    if media_content:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [media_content]
                        })
                
                elif file_type.startswith("video/"):
                    # Process as video
                    media_content = handle_media_upload(uploaded_file)
                    if media_content:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [media_content]
                        })
                
                elif file_type == "application/pdf":
                    # Process as PDF
                    pdf_text = extract_text_from_pdf(uploaded_file, google_api_key)
                    if pdf_text.strip():
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": f"Extracted from PDF:\n\n{pdf_text}"
                            }]
                        })
                
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # Process as DOCX
                    docx_text = extract_text_from_docx(uploaded_file)
                    if docx_text.strip():
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": f"Extracted from Word Document:\n\n{docx_text}"
                            }]
                        })
                
                st.rerun()

        # Camera option below file uploader
        with st.popover("📸 Camera"):
            activate_camera = st.checkbox("Activate camera")
            if activate_camera:
                camera_img = st.camera_input(
                    "Take a picture", 
                    key="camera_img",
                )
                if camera_img:
                    # Process the camera image
                    file_bytes = camera_img.getvalue()
                    file_hash = get_file_hash(file_bytes)
                    
                    if file_hash not in st.session_state.processed_files:
                        st.session_state.processed_files.add(file_hash)
                        
                        # Save the image
                        image = Image.open(BytesIO(file_bytes))
                        img_str = get_image_base64(image)
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                            }]
                        })
                        st.rerun()

        # Audio recorder
        st.write("**🎤 Record Audio**")
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=44100,
            neutral_color="#6ca395"
        )
        
        if audio_bytes:
            audio_hash = get_file_hash(audio_bytes)
            if audio_hash not in st.session_state.processed_files:
                st.session_state.processed_files.add(audio_hash)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    st.session_state.temp_files[f"audio_{audio_hash}"] = tmp_file.name
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "audio_file",
                            "audio_file": tmp_file.name
                        }]
                    })
                    st.rerun()

        # URL input
        url = st.text_input("🌐 Enter URL to summarize")
        if url:
            url_hash = hash(url)
            if url_hash not in st.session_state.processed_files:
                st.session_state.processed_files.add(url_hash)
                url_text = extract_text_from_url(url)
                if url_text.strip():
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"Content from URL {url}:\n\n{url_text[:5000]}..."
                        }]
                    })
                    st.rerun()

        # Download transcript
        st.download_button(
            label="📥 Download Transcript",
            data=download_chat_transcript(),
            file_name="vyomniverse_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    

    # Chat input
    if prompt := st.chat_input("Hi! Ask me anything..."):
        st.session_state.messages.append({
            "role": "user", 
            "content": [{
                "type": "text",
                "text": prompt,
            }]
        })
        st.rerun()

    # Generate response if last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(model_params, google_api_key))

def run_app():
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    run_app()