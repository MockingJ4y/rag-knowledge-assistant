# Industrial RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with React, Vite, and Tailwind CSS. This application allows users to upload documents, process them into chunks, and query an AI assistant for contextual answers based on the uploaded content.

## Features

- **Document Upload**: Support for multiple file formats (PDF, TXT, DOCX, etc.)
- **Intelligent Chunking**: Automatic text chunking with configurable overlap for better context preservation
- **Vector Embeddings**: Local embedding generation for semantic search
- **AI-Powered Responses**: Integration with free AI models via OpenRouter API
- **Real-time Chat Interface**: Interactive Q&A with source citations
- **Statistics Dashboard**: Track document processing metrics
- **Responsive Design**: Modern UI built with Tailwind CSS

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS
- **AI Integration**: OpenRouter API (free models)
- **Icons**: Lucide React
- **Build Tools**: Vite, PostCSS, Autoprefixer

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- OpenRouter API key (free)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MockingJ4y/rag-knowledge-assistant.git
   cd rag-knowledge-assistant
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   - Copy `.env.example` to `.env` (if exists) or create `.env`
   - Add your OpenRouter API key:
     ```
     VITE_OPENROUTER_API_KEY=your-api-key-here
     ```

4. **Get OpenRouter API Key**
   - Visit [OpenRouter.ai](https://openrouter.ai/)
   - Sign up for a free account
   - Generate an API key
   - Add it to your `.env` file

## Usage

1. **Start the development server**
   ```bash
   npm run dev
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5173`

3. **Upload Documents**
   - Click "Upload Documents"
   - Select one or more files
   - Wait for processing

4. **Ask Questions**
   - Type your question in the chat input
   - Get AI-powered answers with source citations

## Configuration

### Chunking Settings
- **Chunk Size**: Number of characters per chunk (default: 500)
- **Chunk Overlap**: Overlap between chunks for context (default: 50)

### AI Settings
- **Model**: AI model to use (default: mistralai/mistral-7b-instruct:free)
- **Top K**: Number of relevant chunks to retrieve (default: 3)
- **Temperature**: Response creativity (0.0-1.0, default: 0.7)

## API Integration

The app uses OpenRouter's free tier, which provides access to various AI models without requiring payment. The current setup uses Mistral 7B Instruct, but you can change the model in the settings.

### Supported Models
- `mistralai/mistral-7b-instruct:free`
- `huggingface/zephyr-7b-beta:free`
- `meta-llama/llama-3.2-3b-instruct:free`

## Project Structure

```
src/
├── components/
│   └── IndustrialRAGSystem.jsx  # Main component
├── index.css                    # Global styles
├── main.jsx                     # App entry point
└── App.jsx                      # Root component
```

## Build for Production

```bash
npm run build
npm run preview
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenRouter for providing free AI API access
- React and Vite communities
- Tailwind CSS for styling