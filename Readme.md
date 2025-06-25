**Chat with PDF using LangChain + OpenAI
**A simple and interactive Streamlit app that lets you chat with the content of PDF files. It uses LangChain, OpenAI, and FAISS to create a question-answering chatbot that understands your documents.
________________________________________
Features
•	 Intelligent Q&A over PDF content
•	 Upload and process multiple PDF files
•	 Vector embeddings using OpenAI + FAISS
•	 Conversational memory with context
•	 Powered by LangChain and OpenAI's GPT models
•	 Simple Streamlit-based UI
________________________________________
________________________________________
________________________________________
Folder Structure
bash
CopyEdit
chat-with-pdf/
│
├── app.py                 # Main Streamlit app
├── .env                  # Contains OpenAI API key
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
________________________________________
How It Works
1.	You upload one or more PDF files.
2.	The app extracts and splits the text into chunks.
3.	Each chunk is embedded into a vector using OpenAI embeddings.
4.	The chunks are stored in a FAISS vector store.
5.	A ConversationalRetrievalChain powered by LangChain retrieves and answers your questions with memory of prior exchanges.
________________________________________
Requirements
•	Python 3.8+
•	OpenAI API Key
•	Internet connection (for API calls)
________________________________________ 
Tech Stack
•	Streamlit
•	LangChain
•	OpenAI
•	FAISS
•	PyPDF2

