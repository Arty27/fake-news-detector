# 🚨 Fake News Detector UI

A beautiful, responsive React dashboard for the Fake News Detection API. Built with Vite, React, and Material-UI.

## ✨ **Features**

- 🎨 **Beautiful Design**: Modern, gradient-based UI with smooth animations
- 📱 **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- 🔍 **Text Analysis**: Analyze articles by pasting text directly
- 🌐 **URL Analysis**: Analyze articles from URLs with automatic content extraction
- 📊 **Comprehensive Results**: Beautiful visualization of all analysis factors
- 🚀 **Real-time Processing**: Live analysis with progress indicators
- 🎯 **Dashboard Ready**: All metrics displayed in an intuitive, professional format

## 🏗️ **Tech Stack**

- **Frontend Framework**: React 18
- **Build Tool**: Vite
- **UI Library**: Material-UI (MUI) v5
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Styling**: Emotion (CSS-in-JS)
- **Routing**: React Router DOM

## 🚀 **Quick Start**

### **1. Install Dependencies**

```bash
npm install
```

### **2. Start Development Server**

```bash
npm run dev
```

````

## 🔧 **Configuration**

### **API Endpoint**
The dashboard connects to the Fake News Detection API at `http://localhost:8000`. To change this:

1. Update the API base URL in each component
2. Or create an environment variable in `.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
````

### **Environment Variables**

Create a `.env` file in the root directory:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_TITLE=Fake News Detector
```
