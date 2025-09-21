# Product Requirements Document (PRD)  
**Product:** AI Meeting Notes & Action Items Extractor  
**Author:** Jana Madkour  
**Date:** September 2025  

---

## 1. Vision  
Enable students, teams, and professionals to instantly transform raw 
meeting transcripts into structured summaries and clear action items, 
reducing time spent on note-taking and improving accountability.  

---

## 2. Goals & Objectives  
- Automatically extract **key points** and **to-do items** from uploaded 
meeting notes.  
- Provide a **simple Streamlit web interface** for uploading text and 
viewing structured summaries.  
- Demonstrate the integration of **NLP (spaCy)** and **TF-IDF ranking** in 
a lightweight app.  

---

## 3. Key Features  
1. Upload plain text meeting notes.  
2. Automatic NLP-based parsing to identify tasks, owners, and deadlines.  
3. Display a structured output with:  
   - **Summary** (high-level meeting recap).  
   - **Action Items** (who needs to do what).  
4. Export results (e.g., copy-to-clipboard or download as .txt).  

---

## 4. Success Metrics (KPIs)  
- **Accuracy:** At least 80% of action items correctly identified.  
- **Usability:** Users should get results in < 5 seconds after uploading 
notes.  
- **Adoption:** Positive feedback from demo users (clarity & ease-of-use).  

---

## 5. User Stories  
- *As a student*, I want to upload class meeting notes so I can quickly 
see assigned tasks.  
- *As a project manager*, I want action items extracted so I can assign 
responsibilities.  
- *As a developer*, I want a modular design so features can be extended 
(e.g., export to Trello).  

---

## 6. Roadmap  
- **MVP (Current):** Upload `.txt` → Summarize → Show action items.  
- **Future:**  
  - Integration with Google Docs / Zoom transcripts.  
  - Export to task managers (Trello, Jira, Asana).  
  - Advanced NLP (transformer-based models for better accuracy).  

---


