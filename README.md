# 🎓 European Master's Thesis Drafting AI

This project is an AI-powered academic assistant designed to help graduate students draft a European-level Master's thesis. By utilizing **Retrieval-Augmented Generation (RAG)**, the application analyzes uploaded academic papers (PDFs) and drafts the thesis section-by-section according to a user-defined outline, generating in-text citations based strictly on the uploaded materials.

## Features
* **PDF Document Processing:** Upload multiple research papers.
* **Context-Aware AI:** Uses OpenAI's LLMs to synthesize information strictly from your uploaded literature.
* **Automated Citations:** References the filenames of the papers it used to generate the text.
* **Section-by-Section Generation:** Bypasses token limits by writing the thesis chapter-by-chapter.
* **Markdown Export:** Download the generated draft in `.md` format.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/thesis-creator-ai.git
   cd thesis-creator-ai