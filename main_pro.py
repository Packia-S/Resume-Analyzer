import os

# FIX: Prevent HuggingFace from creating symlinks on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import streamlit as st
from PIL import Image
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from schema import Profile
from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
import docling
import pandas as pd
from dotenv import load_dotenv
import tempfile

load_dotenv()

icon = Image.open("logo.png")

st.set_page_config(
    page_title="VRNeXGen",
    page_icon=icon,
    layout="wide",
)

st.markdown(
    """
    <h1 style='font-size: 46px; display: flex; align-items: center;'>
        <span style='color:#800000;'>VR</span>NeXGen
    </h1>
    <h8>Modernize 🔺 Automate 🔺 Innovate</h8>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
.box {
    padding: 18px;
    border-radius: 12px;
    margin: 15px 0;
    background: #f1f5f9;
    border-left: 5px solid #2563eb;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
.title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ===================== DUPLICATE POPUP =======================
@st.dialog("Duplicate Email Found")
def show_overwrite_dialog(email, data, csv_file):
    st.write(f"An entry with email **{email}** already exists.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Overwrite Existing"):
            df = pd.read_csv(csv_file)
            df = df[df["email_id"] != email]  
            df_new = pd.DataFrame([data])
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(csv_file, index=False)
            st.success("Resume overwritten successfully!")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.warning("Process cancelled.")
            st.rerun()


# ===================== MAIN TABS =======================
tab1, tab2 = st.tabs(["📃Resume Upload", "📋Filter Resume"])

csv_file = "resume_output.csv"


# ===================== TAB 1: RESUME UPLOAD =======================
with tab1:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

    if uploaded_file:
        st.success("📄 Resume Uploaded")

        if st.button("Convert"):
            # -------- TEMP SAVE ------------
            with st.spinner("Extracting Information..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = tmp.name

                # -------- DOC EXTRACT ------------
                loader = DoclingLoader(
                    temp_path,
                    export_type=ExportType.MARKDOWN,
                    pipeline_options={"do_ocr": False}
                )

                docs = loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.error("❌ Could not extract text from the resume.")
                    st.stop()

                resume_text = docs[0].page_content

            # -------- LLM PROCESSING ------------
            with st.spinner("Generating Insights..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0,
                    google_api_key=settings.google_api_key,
                )

                structured_llm = llm.with_structured_output(schema=Profile)
                response = structured_llm.invoke(resume_text)

                if response is None:
                    st.error("❌ LLM returned no structured data.")
                    st.stop()

                try:
                    data = response.model_dump()
                except Exception as e:
                    st.error(f"❌ Failed to parse structured output: {e}")
                    st.json(response)
                    st.stop()

            # -------- FIX TECHNICAL SKILLS ----------
            tech = data.get("technical_skills", [])
            if isinstance(tech, list) and tech:
                tech = tech[0]
            else:
                tech = {}

            prog = tech.get("programming_languages", [])
            libs = tech.get("libraries_or_frameworks") or tech.get("libraries_or_framework", [])
            tools = tech.get("other_tools", [])

            data["skills"] = list(set(prog + libs + tools))

            # -------- DISPLAY OUTPUT ----------
            st.markdown(f"""
            <div class="box">
                <div class="title">👤 Personal Details</div>
                <p><b>Name:</b> {data.get('fullname')}</p>
                <p><b>Email:</b> {data.get('email_id')}</p>
                <p><b>Phone:</b> {data.get('phone_number')}</p>
                <p><b>Designation:</b> {data.get('designation')}</p>
                <p><b>Location:</b> {data.get('current_location')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["resume_data"] = data
            os.remove(temp_path)

    # -------- SAVE TO CSV ----------
    if "resume_data" in st.session_state:
        if st.button("Save"):
            data = st.session_state["resume_data"]

            df_new = pd.DataFrame([data])

            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)

                if not df[df["email_id"] == data["email_id"]].empty:
                    show_overwrite_dialog(data["email_id"], data, csv_file)
                else:
                    df_new.to_csv(csv_file, mode='a', header=False, index=False)
                    st.success("Resume saved!")
                    st.rerun()
            else:
                df_new.to_csv(csv_file, index=False)
                st.success("Resume saved!")
                st.rerun()


# ===================== TAB 2: FILTER =======================
with tab2:
    if not os.path.exists(csv_file):
        st.warning("No resumes uploaded yet.")
        st.stop()

    df = pd.read_csv(csv_file)

    st.title("🔍 Skills Filtering")

    skill_data = df["skills"].apply(lambda x: eval(x) if isinstance(x, str) else x)

    skill_list = []
    for s in skill_data:
        for item in s:
            skill_list.append(item)

    all_skills = sorted(set(skill_list))

    selected = st.multiselect("Select one or more skills", all_skills)

    if not selected:
        filtered_df = df
    else:
        filtered_df = df[df["skills"].apply(lambda s: all(k in s for k in selected))]

    st.dataframe(filtered_df)
