import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import streamlit as st
from PIL import Image
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from schema import Profile
from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

icon = Image.open("logo.png")

st.set_page_config(
    page_title="VRNeXGen",
    page_icon=icon,
    layout="wide",
)

st.markdown("""
<h1 style='font-size: 46px; display: flex; align-items: center;'>
    <span style='color:#800000;'>VR</span>NeXGen
</h1>
<h8>Modernize 🔺 Automate 🔺 Innovate</h8>
""", unsafe_allow_html=True)

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

@st.dialog("Duplicate Email Found")
def show_overwrite_dialog(email, data, csv_file):
    st.write(f"An entry with email **{email}** already exists in the database.")
    st.write("What would you like to do?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Overwrite Existing"):
            df = pd.read_csv(csv_file)
            df = df[df["email_id"] != email]
            df_new = pd.DataFrame([data])
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(csv_file, index=False)
            st.success(f"Your resume for {email} has been overwritten successfully!")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.warning("The current process has been cancelled.")
            st.rerun()


tab1, tab2 = st.tabs(["📃Resume Upload", "📋Filter Resume"])
csv_file = "resume_output.csv"

with tab1:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file:
        st.success("📄Resume Uploaded")
        if st.button("Convert"):
            with st.spinner("Extracting Information..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # FIXED: DoclingLoader with keyword argument
                loader = DoclingLoader(
                    file_path=temp_path,
                    export_type=ExportType.MARKDOWN,
                    pipeline_options={
                        "do_ocr": False,
                        "do_table_structure": True,
                        "do_layout": True,
                    }
                )

                docs = loader.load()
                resume_text = docs[0].page_content
                

            with st.spinner("Generating Insights..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0,
                    google_api_key=settings.google_api_key,
                )

                structured_llm = llm.with_structured_output(schema=Profile)
                response = structured_llm.invoke(resume_text)

            data = response.model_dump()

            tech = data.get("technical_skills", [{}])[0]
            
            all_skills = (
                tech.get("programming_languages", []) +
                tech.get("libraries_or_frameworks", []) +
                tech.get("other_tools", []) +
                data.get('interpersonal_skills', [])
            )
            
            # Add the new 'skills' column to the dictionary
            data["skills"] = all_skills

            # Display Personal Details
            st.markdown(f"""
            <div class="box">
                <div class="title">👤 Personal Details</div>
                <p><b>Name:</b> {data.get('fullname')}</p>
                <p><b>Email:</b> {data.get('email_id')}</p>
                <p><b>Phone:</b> {data.get('phone_number')}</p>
                <p><b>Designation:</b> {data.get('designation')}</p>
                <p><b>Current Location:</b> {data.get('current_location')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Technical Skills
            # tech = data.get("technical_skills", [{}])[0]
            prog_langs = tech.get("programming_languages", [])
            frameworks = tech.get("libraries_or_frameworks", [])
            tools = tech.get("other_tools", [])

            st.markdown(f"""
            <div class="box">
                <div class="title">💻 Programming Languages</div>
                {"".join([f"<span style='background:#e0e7ff;padding:6px 12px;margin:4px;border-radius:8px;'>{lang}</span>" for lang in prog_langs])}
            </div>
            <div class="box">
                <div class="title">💻 Libraries or Frameworks</div>
                {"".join([f"<span style='background:#e0e7ff;padding:6px 12px;margin:4px;border-radius:8px;'>{fw}</span>" for fw in frameworks])}
            </div>
            <div class="box">
                <div class="title">💻 Other Tools</div>
                {"".join([f"<span style='background:#e0e7ff;padding:6px 12px;margin:4px;border-radius:8px;'>{t}</span>" for t in tools])}
            </div>
            <div class="box">
                <div class="title">💻 Interpersonal Skills</div>
                {"".join([f"<span style='background:#e0e7ff;padding:6px 12px;margin:4px;border-radius:8px;'>{s}</span>" for s in data.get('interpersonal_skills', [])])}
            </div>
            """, unsafe_allow_html=True)

          
            st.markdown(f"""
            <div class="box">
                <div class="title">👩🏻‍💻 Working Details</div>
                <p><b>Year of Experience:</b> {data.get('year_of_experience')}</p>
                <p><b>Current CTC:</b> {data.get('current_ctc')}</p>
                <p><b>Current Company:</b> {data.get('current_company')}</p>
                <p><b>Expected CTC:</b> {data.get('expected_ctc')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="box">
                <div class="title">🌐 Links</div>
                <p><b>LinkedIn URL:</b> {data.get('linkedin_url')}</p>
                <p><b>GitHub URL:</b> {data.get('github_url')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="box">
                <div class="title">📄 Certifications</div>
                <p><b>Certifications:</b> {data.get('certifications')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="box">
                <div class="title">📃 Summary</div>
                <p><b>Summary:</b> {data.get('summary')}</p>
            </div>
            """, unsafe_allow_html=True)
  
            st.markdown(f"""
            <div class="box">
                <div class="title">👩‍💻 Portfolio </div>
                <p><b>Portfolio Project URL:</b> {data.get('portfolio_project_url')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["resume_data"] = data
            os.remove(temp_path)

    if "resume_data" in st.session_state:
        if st.button("Save"):
            data = st.session_state["resume_data"]
            df_new = pd.DataFrame([data])
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if not df[df['email_id'] == data['email_id']].empty:
                    show_overwrite_dialog(data['email_id'], data, csv_file)
                else:
                    df_new.to_csv(csv_file, mode='a', header=False, index=False)
                    st.success("📄Resume data successfully saved")
                    st.rerun()
            else:
                df_new.to_csv(csv_file, index=False)
                st.success("📄Resume data successfully saved")


with tab2:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        st.title("🔍 Skills Filtering")
        try:
          skill_data = df["skills"].apply(lambda x: eval(x) if isinstance(x, str) else x)
          skill_list = [skill for skills in skill_data for skill in skills]
          all_skill_details = sorted(set(skill_list))
          selected_skills = st.multiselect("Select one or more skills", all_skill_details)
          if selected_skills:
              filtered_df = df[df['skills'].apply(lambda skill_set: all(skill in skill_set for skill in selected_skills))]
          else:
              filtered_df = df
        except Exception as e:
          st.error(f"Error reading skills data. Please ensure the data in `resume_output.csv` is correctly formatted: {e}")
    else:
        st.info("The `resume_output.csv` file has not been created yet. Please upload and save a resume in the 'Resume Upload' tab first.")
            # st.dataframe(filtered_df)








