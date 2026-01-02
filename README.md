# NeuroMind AI - Advanced Memory Chatbot

A sophisticated AI chatbot with advanced memory capabilities and a professional web interface, powered by episodic and semantic memory systems.

## ✨ Features

### 🤖 Advanced AI Capabilities

- **Episodic Memory**: Remembers specific conversations and their context with timestamps
- **Semantic Memory**: Stores and retrieves factual knowledge from conversations
- **Short-term Memory**: Maintains context within conversation sessions
- **Contextual Responses**: AI responses informed by relevant past interactions

### 🎨 Professional UI/UX

- **Modern Design**: Clean, professional interface with smooth animations
- **Dark/Light Mode**: Toggle between themes with persistent preferences
- **Responsive Layout**: Optimized for desktop, tablet, and mobile devices
- **Real-time Updates**: Live typing indicators and instant message delivery
- **Accessibility**: WCAG compliant with keyboard navigation and screen reader support

### 📊 Analytics & Insights

- **Memory Visualization**: Real-time display of retrieved memories
- **Performance Metrics**: Response times, memory hit counts, session duration
- **Message Statistics**: Track conversation metrics and patterns
- **Export Functionality**: Download chat history as text files

### ⚙️ Advanced Settings

- **Customizable Memory Limits**: Adjust how many memories to retrieve
- **Notification Preferences**: Sound alerts and visual feedback options
- **Auto-scroll Control**: Toggle automatic scrolling to new messages
- **Timestamp Display**: Show/hide message timestamps

### 🎯 User Experience

- **Voice Input**: Speech-to-text functionality (where supported)
- **Message History**: Navigate through previous messages with arrow keys
- **Keyboard Shortcuts**: Quick actions with Ctrl/Cmd key combinations
- **Toast Notifications**: Non-intrusive status updates
- **Loading States**: Visual feedback during AI processing

## 🚀 Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Application**:

   ```bash
   python app.py
   ```

4. **Access the Interface**:
   Open `http://127.0.0.1:5000` in your browser

## 🏗️ Architecture

### Backend (Flask/Python)

- **Web Framework**: Flask with RESTful API endpoints
- **AI Integration**: Groq API (Llama 3.3 70B model)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: HNSW (Hierarchical Navigable Small World)
- **Memory Systems**: Custom episodic and semantic memory implementations

### Frontend (HTML/CSS/JavaScript)

- **Framework**: Vanilla JavaScript with modern ES6+ features
- **Styling**: Custom CSS with CSS Variables for theming
- **Icons**: Font Awesome 6 for consistent iconography
- **Typography**: Inter font for professional appearance

### Memory Architecture

```
Episodic Memory: Conversation instances with full context
├── User input
├── AI response
├── Timestamp
└── Embedding vector

Semantic Memory: Factual knowledge extracted from conversations
├── Knowledge statements
└── Embedding vectors

Short-term Memory: Recent conversation context
└── Last K message pairs
```

## 📡 API Endpoints

### `GET /`

Serves the main chat interface

### `POST /chat`

Send a message and receive AI response with memory data

```json
{
  "message": "Hello, how are you?",
  "memory_limit": 3
}
```

**Response**:

```json
{
  "response": "I'm doing well, thank you for asking!",
  "episodic_hits": [...],
  "semantic_hits": [...],
  "processing_time": 1250.5,
  "memory_count": 2,
  "timestamp": 1735689600.0
}
```

### `GET /stats`

Retrieve system statistics

```json
{
  "episodic_memory_count": 45,
  "semantic_memory_count": 23,
  "total_conversations": 12
}
```

## 🎹 Keyboard Shortcuts

- `Ctrl/Cmd + K`: Focus message input
- `Ctrl/Cmd + L`: Toggle theme
- `Ctrl/Cmd + /`: Open settings
- `↑/↓`: Navigate message history
- `Enter`: Send message (Shift+Enter for new line)

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)

### Memory Configuration

- **Episodic Memory**: Stores up to 10,000 episodes
- **Semantic Memory**: Stores up to 5,000 concepts
- **Short-term Memory**: Maintains last 3 message pairs

### Performance Tuning

- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **HNSW Parameters**: ef_construction=200, M=16
- **Similarity Metric**: Cosine similarity

## 🌟 Advanced Features

### Memory Learning

The system automatically learns from conversations:

1. **Episodic Storage**: Every conversation turn is stored with full context
2. **Semantic Extraction**: AI identifies and stores factual knowledge
3. **Context Retrieval**: Relevant memories are retrieved for each query

### Intelligent Responses

- **Context Awareness**: Responses consider conversation history
- **Memory Integration**: Past experiences inform current responses
- **Adaptive Learning**: System improves responses over time

### Professional Interface

- **Status Indicators**: Real-time connection and processing status
- **Progress Feedback**: Loading states and processing indicators
- **Error Handling**: Graceful error messages and recovery
- **Data Export**: Conversation history preservation

## 📱 Browser Support

- **Chrome/Edge**: Full feature support including voice input
- **Firefox**: Full feature support
- **Safari**: Full feature support (iOS 14.5+ for voice input)
- **Mobile Browsers**: Responsive design with touch optimization

## 🔒 Security & Privacy

- **Local Processing**: All memory data stored locally
- **No Data Collection**: Conversations remain on your device
- **API Security**: Secure communication with AI services
- **Input Validation**: Sanitized user inputs and outputs

## 🚀 Deployment

### Local Development

```bash
python app.py
```

### Production Deployment

```bash
export FLASK_ENV=production
python app.py
```

### Docker Support

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq**: For providing fast AI inference
- **Sentence Transformers**: For high-quality embeddings
- **HNSW**: For efficient vector similarity search
- **Font Awesome**: For beautiful icons
- **Inter Font**: For professional typography

---

**Built with ❤️ using Python, Flask, and modern web technologies**
