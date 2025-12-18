import streamlit as st
import tempfile
import os
from PIL import Image
from models.embedder import CLIPEmbedder
from models.scorer import compatibility_score

# Initialize embedder (cached to avoid reloading on every interaction)
@st.cache_resource
def load_embedder():
    return CLIPEmbedder()

embedder = load_embedder()

st.set_page_config(page_title="FitCheck 👗", page_icon="👗")

st.markdown(
    """
    # 👗 FitCheck  
    **Beginner-friendly fashion compatibility demo**  
    Upload 2–5 clothing items and see how well they go together.
    """
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload Outfit Items",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded**")
    
    # Display uploaded images
    cols = st.columns(min(len(uploaded_files), 5))
    for idx, uploaded_file in enumerate(uploaded_files[:5]):
        with cols[idx]:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption=f"Item {idx + 1}")

# Button to check compatibility
if st.button("Check Fit ✨", type="primary"):
    if uploaded_files is None or len(uploaded_files) < 2:
        st.error("Please upload at least 2 items!")
    else:
        with st.spinner("Analyzing outfit compatibility..."):
            # Save uploaded files temporarily
            temp_paths = []
            try:
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_paths.append(tmp_file.name)
                
                # Get embeddings
                embeddings = embedder.encode_images(temp_paths)
                score = compatibility_score(embeddings)
                
                # Display results
                st.success(f"**Compatibility Score: {score}/100**")
                st.progress(score / 100, text=f"{score:.1f}% compatible")
                
            finally:
                # Clean up temporary files
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
