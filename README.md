# Nexus Search 
```Agentic Search Engine```

<div align="center">

  <img src="https://github.com/manumishra12/Nexus-Search/blob/main/assets/dark%20mode.png" alt="NEXUS SEARCH" width="900" height="500">
  
  <br>
  
  <img src="https://github.com/manumishra12/Nexus-Search/blob/main/assets/light%20mode.png" alt="NEXUS SEARCH" width="900" height="500">

  <h2>Video Demonstration</h2>
  <video width="320" height="240" controls>
    <source src="https://github.com/manumishra12/Nexus-Search/blob/main/assets/nexus%20_search.mp4?raw=true" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <p>You can also download the video directly <a href="https://github.com/manumishra12/Nexus-Search/blob/main/assets/nexus%20_search.mp4?raw=true">here</a>.</p>

  <br>

  <h2>Download PDF</h2>
  <a href="https://github.com/manumishra12/Nexus-Search/blob/main/assets/documentation.pdf?raw=true" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/6/6c/PDF_icon.svg" alt="PDF Icon" width="30" height="40"> Download Documentation
  </a>

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Frontend Components](#frontend-components)
- [Usage Examples](#usage-examples)
- [Multi-Agent Workflow](#multi-agent-workflow)
- [File Processing](#file-processing)
- [Document Generation](#document-generation)
- [Development](#development)
- [Deployment](#deployment)


## 🎯 Overview

NEXUS SEARCH is a cutting-edge, ultra-premium agentic search engine that combines the power of **LangChain**, **LangGraph**, and **Tavily** to deliver unparalleled search experiences. Built with a sophisticated 13-agent architecture, it provides intelligent content analysis, multi-modal processing, and professional document generation.

### 🌟 What Makes NEXUS AI Unique?

- **🤖 13-Agent Architecture**: Specialized AI agents for different aspects of search and analysis
- **🔍 Advanced Search**: Powered by Tavily with intelligent content validation
- **📄 Multi-Modal Processing**: Handle PDFs, documents, images, and videos seamlessly
- **🎨 Professional UI**: Modern React interface with dark/light modes
- **📊 Credibility Scoring**: AI-powered source validation and fact-checking
- **📑 PDF Generation**: Create professional research reports automatically

## ✨ Features

### 🔍 **Advanced Search Capabilities**
- **Multi-Engine Search**: Web, Academic, News, Images, Videos
- **Intelligent Summarization**: AI-powered content analysis
- **Real-Time Processing**: Live updates on search progress
- **Follow-Up Questions**: AI-generated exploration suggestions

### 🤖 **Multi-Agent Architecture**
- **Multimodal Processor**: Handles uploaded files and context
- **Query Analyzer**: Understands intent and complexity
- **Search Strategist**: Plans optimal search approach
- **Content Validator**: Scores credibility and quality
- **Fact Checker**: Validates information accuracy
- **Synthesis Expert**: Creates comprehensive answers
- **Quality Assurance**: Ensures output excellence

### 📄 **Document Processing**
- **File Upload**: Support for PDF, DOC, TXT, Images
- **Text Extraction**: OCR and document parsing
- **Thumbnail Generation**: Visual previews
- **Content Analysis**: Intelligent metadata extraction

### 📊 **Professional Reporting**
- **PDF Generation**: Create detailed research reports
- **Custom Sections**: Configurable document structure
- **Media Integration**: Include images and charts
- **Citation Management**: Automatic source attribution

### 🎨 **Premium User Interface**
- **Modern Design**: Clean, professional interface
- **Dark/Light Mode**: Customizable themes
- **Responsive Layout**: Works on all devices
- **Real-Time Updates**: Live search progress indicators

## 🏗️ Architecture

```mermaid
graph TB
    A[Frontend - React] --> B[FastAPI Backend]
    B --> C[LangGraph Workflow]
    C --> D[13 Specialized Agents]
    B --> E[Tavily Search Engine]
    B --> F[Gemini LLM]
    B --> G[File Processing]
    B --> H[PDF Generation]
    
    D --> D1[Multimodal Processor]
    D --> D2[Query Analyzer]
    D --> D3[Search Strategist]
    D --> D4[Enhanced Search]
    D --> D5[Content Validator]
    D --> D6[Media Extractor]
    D --> D7[Fact Checker]
    D --> D8[Synthesis Expert]
    D --> D9[Multimodal Synthesizer]
    D --> D10[Summarization]
    D --> D11[Citation Specialist]
    D --> D12[Insight Generator]
    D --> D13[Quality Assurance]
```

## 🛠️ Technology Stack

### **Backend**
- **Framework**: FastAPI 0.104+
- **AI/ML**: LangChain, LangGraph, Google Gemini
- **Search**: Tavily Search API
- **File Processing**: PyPDF2, python-docx, Pillow, pytesseract
- **PDF Generation**: ReportLab
- **Database**: In-memory (production: PostgreSQL/MongoDB)

### **Frontend**
- **Framework**: React 18+
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **State Management**: React Hooks
- **HTTP Client**: Fetch API

### **Infrastructure**
- **CORS**: FastAPI CORS middleware
- **File Storage**: Local filesystem (production: AWS S3/GCS)
- **Environment**: Python 3.8+, Node.js 16+

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nexus-ai.git
cd nexus-ai
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export TAVILY_API_KEY="your_tavily_api_key"

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs

## 📦 Installation

### Backend Dependencies

Create a `requirements.txt` file:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
langchain==0.0.345
langgraph==0.0.20
langchain-google-genai==0.0.6
langchain-community==0.0.10
tavily-python==0.3.0
requests==2.31.0
beautifulsoup4==4.12.2
PyPDF2==3.0.1
python-docx==1.1.0
pandas==2.1.4
numpy==1.24.3
Pillow==10.1.0
pytesseract==0.3.10
opencv-python==4.8.1.78
moviepy==1.0.3
speech-recognition==3.10.0
pydub==0.25.1
reportlab==4.0.7
psutil==5.9.6
```

### Frontend Dependencies

Create a `package.json` file:

```json
{
  "name": "nexus-ai-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "framer-motion": "^10.16.4",
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys
GEMINI_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_search_api_key

# Server Configuration
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# File Upload Configuration
MAX_FILE_SIZE=104857600  # 100MB
UPLOAD_DIR=uploads
THUMBNAILS_DIR=thumbnails
GENERATED_DOCS_DIR=generated_docs

# Processing Configuration
MAX_SOURCES=12
DEFAULT_TIMEOUT=30
```

### API Keys Setup

1. **Google Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add to environment variables

2. **Tavily Search API Key**:
   - Visit [Tavily](https://tavily.com/)
   - Sign up and get your API key
   - Add to environment variables

## 📚 API Reference

### Search Endpoint

```http
POST /search
```

**Request Body:**
```json
{
  "query": "string",
  "search_mode": "comprehensive",
  "search_type": "web",
  "uploaded_file_ids": ["string"],
  "extract_media": true,
  "max_sources": 10,
  "stream": false
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "id": "string",
      "title": "string",
      "url": "string",
      "snippet": "string",
      "credibility_score": 0.8,
      "domain": "string",
      "is_academic": false
    }
  ],
  "extracted_media": [
    {
      "type": "image",
      "title": "string",
      "url": "string",
      "thumbnail": "string"
    }
  ],
  "follow_up_questions": ["string"],
  "credibility_score": 0.85,
  "processing_time": 2.5
}
```

### File Upload Endpoint

```http
POST /upload
```

**Request:** Multipart form data with file

**Response:**
```json
{
  "file_id": "string",
  "filename": "string",
  "file_type": "pdf",
  "file_size": 1024,
  "upload_time": "2024-01-01T00:00:00Z",
  "extracted_text": "string",
  "thumbnail_url": "string"
}
```

### Document Generation Endpoint

```http
POST /generate-document
```

**Request Body:**
```json
{
  "title": "string",
  "query": "string",
  "sections": [
    {
      "id": 1,
      "title": "string",
      "content": "string",
      "type": "text"
    }
  ],
  "searchResults": {},
  "options": {
    "includeImages": true,
    "includeCharts": true,
    "includeSources": true
  }
}
```

**Response:** PDF file download

## 🧩 Frontend Components

### Core Components

#### `NexusAI` - Main Application Component
- Manages global state and routing
- Handles dark/light mode
- Coordinates search and file operations

#### `SourceCard` - Enhanced Source Display
```jsx
<SourceCard 
  source={sourceData}
  index={0}
  darkMode={true}
  onSummaryRequest={handleSummary}
/>
```

#### `SummaryPanel` - Research Overview
```jsx
<SummaryPanel
  results={searchResults}
  query={searchQuery}
  darkMode={darkMode}
  onGenerateDocument={handleGenerate}
/>
```

#### `DocumentGenerator` - PDF Creation Modal
```jsx
<DocumentGenerator
  isOpen={isOpen}
  onClose={handleClose}
  searchResults={results}
  query={query}
  darkMode={darkMode}
/>
```

#### `FileCard` - File Display Component
```jsx
<FileCard
  file={fileData}
  darkMode={darkMode}
  onDelete={handleDelete}
/>
```

### Custom Hooks

#### `useFileUpload` - File Upload Management
```jsx
const { getRootProps, getInputProps, isDragActive } = useFileUpload({
  onDrop: handleFileDrop,
  accept: { 'application/pdf': ['.pdf'] },
  multiple: true
});
```

#### `useCopyToClipboard` - Clipboard Operations
```jsx
const { copied, copyToClipboard } = useCopyToClipboard();
```

## 💡 Usage Examples

### Basic Search

```javascript
// Perform a simple web search
const searchRequest = {
  query: "artificial intelligence trends 2024",
  search_type: "web",
  max_sources: 10
};

const response = await fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(searchRequest)
});

const results = await response.json();
```

### File Upload and Search

```javascript
// Upload file and include in search
const formData = new FormData();
formData.append('file', file);

const uploadResponse = await fetch('/upload', {
  method: 'POST',
  body: formData
});

const fileData = await uploadResponse.json();

// Use file in search
const searchRequest = {
  query: "analyze this document for key insights",
  uploaded_file_ids: [fileData.file_id]
};
```

### Generate PDF Report

```javascript
// Generate comprehensive PDF report
const documentRequest = {
  title: "AI Research Report",
  query: "artificial intelligence",
  sections: [
    { id: 1, title: "Executive Summary", content: "", type: "text" },
    { id: 2, title: "Sources", content: "", type: "sources" }
  ],
  searchResults: results,
  options: {
    includeImages: true,
    includeSources: true
  }
};

const pdfResponse = await fetch('/generate-document', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(documentRequest)
});
```

## 🤖 Multi-Agent Workflow

The 13-agent architecture processes queries through specialized stages:

### Agent Pipeline

1. **Multimodal Processor** - Processes uploaded files and context
2. **Query Analyzer** - Analyzes intent, complexity, and domains
3. **Search Strategist** - Plans optimal search approach
4. **Enhanced Search** - Executes search using Tavily
5. **Content Validator** - Validates and scores content credibility
6. **Media Extractor** - Extracts relevant media from sources
7. **Fact Checker** - Performs fact-checking and verification
8. **Synthesis Expert** - Creates comprehensive answers
9. **Multimodal Synthesizer** - Enhances with multimodal insights
10. **Summarization** - Creates executive summaries
11. **Citation Specialist** - Manages source citations
12. **Insight Generator** - Generates follow-up questions
13. **Quality Assurance** - Final quality check and scoring

### Workflow Configuration

```python
# Example agent configuration
workflow = StateGraph(AdvancedAgentState)

# Add all agents
workflow.add_node("multimodal_processor", multimodal_processor_agent)
workflow.add_node("query_analyzer", query_analyzer_agent)
# ... other agents

# Define execution flow
workflow.set_entry_point("multimodal_processor")
workflow.add_edge("multimodal_processor", "query_analyzer")
# ... other edges
```

## 📄 File Processing

### Supported File Types

| Type | Extensions | Features |
|------|------------|----------|
| **PDF** | `.pdf` | Text extraction, thumbnail generation |
| **Documents** | `.doc`, `.docx` | Content parsing, metadata extraction |
| **Text** | `.txt` | Direct content reading |
| **Images** | `.png`, `.jpg`, `.jpeg`, `.gif` | OCR text extraction, thumbnails |
| **Videos** | `.mp4`, `.avi`, `.mov` | Metadata extraction (future: transcription) |

### Processing Pipeline

```python
# Example file processing
async def process_file(file_content, file_type):
    if file_type == "pdf":
        return await FileProcessor.process_pdf(file_content)
    elif file_type == "image":
        return await ImageProcessor.process_image(file_content)
    # ... other processors
```

## 📊 Document Generation

### PDF Features

- **Professional Layout**: Multi-column layouts with proper typography
- **Table of Contents**: Automatic generation with page numbers
- **Source Citations**: Properly formatted academic citations
- **Media Integration**: Embedded images and charts
- **Custom Sections**: Configurable document structure
- **Metadata**: Document properties and generation info

### Document Structure

```python
# Example document sections
sections = [
    {"id": 1, "title": "Executive Summary", "type": "text"},
    {"id": 2, "title": "Key Findings", "type": "text"},
    {"id": 3, "title": "Source Analysis", "type": "sources"},
    {"id": 4, "title": "Media Gallery", "type": "media"},
    {"id": 5, "title": "Conclusion", "type": "text"}
]
```

## 🔧 Development

### Project Structure

```
nexus-ai/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── uploads/               # File upload directory
│   ├── thumbnails/            # Generated thumbnails
│   └── generated_docs/        # Generated PDF reports
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── index.js           # React entry point
│   │   └── index.css          # Tailwind styles
│   ├── public/                # Static assets
│   └── package.json           # Node dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

### Development Setup

1. **Backend Development**:
   ```bash
   cd backend
   uvicorn main:app --reload --log-level debug
   ```

2. **Frontend Development**:
   ```bash
   cd frontend
   npm start
   ```

3. **Testing**:
   ```bash
   # Backend tests
   pytest tests/

   # Frontend tests
   npm test
   ```

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **JavaScript**: Use ESLint and Prettier
- **Commits**: Conventional Commits format

## 🚀 Deployment

### Production Environment

1. **Backend Deployment**:
   ```bash
   # Using Docker
   docker build -t nexus-ai-backend .
   docker run -p 8000:8000 nexus-ai-backend

   # Using Gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Frontend Deployment**:
   ```bash
   # Build for production
   npm run build

   # Deploy to static hosting (Vercel, Netlify, etc.)
   npm install -g vercel
   vercel --prod
   ```

### Environment Variables (Production)

```env
# Production API Keys
GEMINI_API_KEY=prod_gemini_key
TAVILY_API_KEY=prod_tavily_key

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# File Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_BUCKET_NAME=nexus-ai-files

# Security
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=["yourdomain.com"]
```

### Docker Configuration

**Dockerfile (Backend)**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    volumes:
      - ./uploads:/app/uploads

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

## 🧪 Testing

### Backend Tests

```python
# tests/test_search.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_search_endpoint():
    response = client.post("/search", json={
        "query": "test query",
        "search_type": "web"
    })
    assert response.status_code == 200
    assert "answer" in response.json()

def test_file_upload():
    with open("test_file.pdf", "rb") as f:
        response = client.post("/upload", files={"file": f})
    assert response.status_code == 200
```

### Frontend Tests

```javascript
// src/App.test.js
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders NEXUS AI title', () => {
  render(<App />);
  const titleElement = screen.getByText(/NEXUS AI/i);
  expect(titleElement).toBeInTheDocument();
});

test('search functionality', () => {
  render(<App />);
  const searchInput = screen.getByPlaceholderText(/ask me anything/i);
  expect(searchInput).toBeInTheDocument();
});
```

## 📈 Performance Optimization

### Backend Optimizations

- **Async Processing**: All I/O operations are asynchronous
- **Caching**: Redis for search result caching
- **Connection Pooling**: Database connection optimization
- **File Compression**: Automatic image and document compression

### Frontend Optimizations

- **Lazy Loading**: Components and images loaded on demand
- **Memoization**: React.memo for expensive components
- **Virtual Scrolling**: For large result sets
- **Code Splitting**: Bundle optimization

## 🔒 Security

### Authentication & Authorization

```python
# Example JWT authentication
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    # Verify JWT token
    return user
```

### File Upload Security

- File type validation
- Size limitations
- Malware scanning
- Secure file storage

### API Security

- Rate limiting
- CORS configuration
- Input validation
- SQL injection prevention

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest` and `npm test`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Contribution Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Use conventional commit messages
- Keep PRs focused and small

### Areas for Contribution

- **New File Processors**: Add support for more file types
- **Additional Search Engines**: Integrate more search APIs
- **UI Improvements**: Enhance user interface components
- **Performance**: Optimize search and processing speed
- **Documentation**: Improve guides and examples

## 📝 Changelog

### v1.0.0 (2024-01-01)
- Initial release
- 13-agent architecture
- Multi-modal file processing
- PDF report generation
- Professional UI with dark/light modes

### v0.9.0 (2023-12-15)
- Beta release
- Core search functionality
- Basic file upload
- Initial agent implementation

## 🆘 Troubleshooting

### Common Issues

**Backend won't start**:
```bash
# Check Python version
python --version  # Should be 3.8+

# Install dependencies
pip install -r requirements.txt

# Check API keys
echo $GEMINI_API_KEY
echo $TAVILY_API_KEY
```

**Frontend build errors**:
```bash
# Clear cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

**File upload issues**:
- Check file size (max 100MB)
- Verify file format support
- Ensure upload directory permissions


## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - AI application framework
- [Tavily](https://tavily.com/) - Search API provider
- [Google Gemini](https://ai.google.dev/) - Large language model
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [React](https://reactjs.org/) - Frontend library
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
