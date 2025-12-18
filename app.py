import streamlit as st
import tempfile
import os
import json
from PIL import Image
from models.embedder import CLIPEmbedder
from models.scorer import compatibility_score
import base64

# Base64 func for image to base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Initialize embedder (cached to avoid reloading on every interaction)
@st.cache_resource
def load_embedder():
    return CLIPEmbedder()

embedder = load_embedder()

st.set_page_config(
    page_title="FitCheck 👗", 
    page_icon="👗",
    layout="wide"
)

st.markdown(
    """
    # 👗 FitCheck  
    **Beginner-friendly fashion compatibility demo**  
    Upload 2–5 clothing items and see how well they go together.
    """
)


col1, col2= st.columns([2,1], vertical_alignment="top")

with col1:
    tabs = st.tabs(["Upload Outfit Items", "Model Suggested Outfits"])
    with tabs[0]:
        st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:left; gap:15px;">
        
        </div>
        """,
        unsafe_allow_html=True
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
    with tabs[1]:
        st.markdown("### 🏆 Top 10 Highly Rated Outfits from Dataset")
        st.markdown("*These outfits have the highest internal compatibility scores.*")
        
        outfits_folder = "outfits"
        
        if not os.path.exists(outfits_folder):
            st.info("💡 Run `python suggest_outfits.py` first to generate the top 10 outfits!")
        else:
            # Get all outfit folders, sorted
            outfit_dirs = []
            if os.path.exists(outfits_folder):
                for item in os.listdir(outfits_folder):
                    item_path = os.path.join(outfits_folder, item)
                    if os.path.isdir(item_path):
                        outfit_dirs.append(item_path)
            
            outfit_dirs.sort()  # Sort to get outfit_01, outfit_02, etc.
            
            if not outfit_dirs:
                st.info("No outfits found. Run `python suggest_outfits.py` to generate them.")
            else:
                # Display each outfit
                for rank, outfit_dir in enumerate(outfit_dirs[:10], 1):
                    st.markdown(f"---")
                    st.markdown(f"#### 🥇 Rank #{rank}: {os.path.basename(outfit_dir)}")
                    
                    # Load reasoning if available
                    reasoning_path = os.path.join(outfit_dir, "reasoning.json")
                    if os.path.exists(reasoning_path):
                        try:
                            with open(reasoning_path, "r") as f:
                                reasoning = json.load(f)
                            
                            # Display reasoning
                            reason_cols = st.columns(2)
                            with reason_cols[0]:
                                if reasoning.get("heuristic_reason"):
                                    st.info(f"🎯 **Heuristic:** {reasoning['heuristic_reason']}")
                            with reason_cols[1]:
                                if reasoning.get("text_reason"):
                                    st.info(f"💭 **Style:** {reasoning['text_reason']}")
                        except Exception as e:
                            st.warning(f"Could not load reasoning: {e}")
                    
                    # Get all images in this outfit folder
                    image_files = [
                        f for f in os.listdir(outfit_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    image_files.sort()
                    
                    if image_files:
                        # Display images in a grid with category labels
                        cols = st.columns(min(len(image_files), 5))
                        for idx, img_file in enumerate(image_files):
                            img_path = os.path.join(outfit_dir, img_file)
                            try:
                                with cols[idx % len(cols)]:
                                    img = Image.open(img_path)
                                    # Extract category from filename (format: category_itemid.jpg)
                                    category = img_file.split('_')[0].title()
                                    st.image(img, use_container_width=True, caption=category)
                            except Exception as e:
                                st.error(f"Error loading {img_file}: {e}")
                    else:
                        st.warning("No images found in this outfit folder.")

with col2:
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
                    
                    # Get embeddings after all files are saved
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
