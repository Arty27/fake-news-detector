import { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Paper,
  Chip
} from '@mui/material'
import { Send, FileText, AlertCircle } from 'lucide-react'
import axios from 'axios'

const TextAnalysis = ({ onAnalysisComplete, onAnalysisStart, onAnalysisEnd, isLoading }) => {
  const [text, setText] = useState('')
  const [error, setError] = useState('')
  const [charCount, setCharCount] = useState(0)

  const handleTextChange = (event) => {
    const newText = event.target.value
    setText(newText)
    setCharCount(newText.length)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    if (text.trim().length < 10) {
      setError('Text must be at least 10 characters long')
      return
    }

    try {
      onAnalysisStart()
      setError('')

      const response = await axios.post('http://localhost:8000/text/analyze', {
        text: text.trim()
      })

      if (response.data.success) {
        // Pass both the analysis results and the original input text
        onAnalysisComplete({
          ...response.data.data,
          original_input: text.trim(),
          input_type: 'text'
        })
      } else {
        setError(response.data.error || 'Analysis failed')
      }
    } catch (err) {
      console.error('Analysis error:', err)
      if (err.response?.data?.error) {
        setError(err.response.data.error)
      } else if (err.code === 'ERR_NETWORK') {
        setError('Unable to connect to the API server. Please make sure the server is running.')
      } else {
        setError('An unexpected error occurred. Please try again.')
      }
    } finally {
      onAnalysisEnd()
    }
  }

  const getCharCountColor = () => {
    if (charCount < 10) return 'error'
    if (charCount < 100) return 'warning'
    if (charCount < 1000) return 'info'
    return 'success'
  }

  const getCharCountLabel = () => {
    if (charCount < 10) return 'Too short'
    if (charCount < 100) return 'Short'
    if (charCount < 1000) return 'Good'
    return 'Long'
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        <FileText size={24} style={{ marginRight: 8, verticalAlign: 'middle' }} />
        Text Article Analysis
      </Typography>

      <Paper 
        elevation={0} 
        sx={{ 
          p: 3, 
          border: '2px dashed',
          borderColor: 'divider',
          borderRadius: 2,
          mb: 3,
          backgroundColor: 'background.paper'
        }}
      >
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Paste or type your article text below. The system will analyze it using multiple AI techniques.
        </Typography>
        
        <TextField
          multiline
          rows={8}
          fullWidth
          variant="outlined"
          placeholder="Enter your article text here... (minimum 10 characters)"
          value={text}
          onChange={handleTextChange}
          disabled={isLoading}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2,
              fontSize: '1rem',
              lineHeight: 1.6
            }
          }}
        />

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={`${charCount} characters`}
              color={getCharCountColor()}
              size="small"
              variant="outlined"
            />
            <Chip
              label={getCharCountLabel()}
              color={getCharCountColor()}
              size="small"
            />
          </Box>
          
          <Button
            variant="contained"
            size="large"
            onClick={handleAnalyze}
            disabled={isLoading || text.trim().length < 10}
            startIcon={isLoading ? <CircularProgress size={20} /> : <Send />}
            sx={{
              borderRadius: 2,
              px: 4,
              py: 1.5,
              fontSize: '1rem',
              fontWeight: 600,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
                transform: 'translateY(-1px)',
                boxShadow: '0 8px 16px rgba(0,0,0,0.2)'
              },
              transition: 'all 0.3s ease'
            }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Text'}
          </Button>
        </Box>
      </Paper>

      {error && (
        <Alert 
          severity="error" 
          icon={<AlertCircle />}
          sx={{ mb: 3, borderRadius: 2 }}
        >
          {error}
        </Alert>
      )}

      {/* Tips Section */}
      <Paper elevation={0} sx={{ p: 3, backgroundColor: 'grey.50', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'primary.main' }}>
          💡 Analysis Tips
        </Typography>
        <Box component="ul" sx={{ pl: 2, mt: 1 }}>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Include specific names, dates, and locations for better entity recognition
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Longer articles (100+ characters) provide more context for accurate analysis
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            The system works best with news articles, social media posts, and factual content
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}

export default TextAnalysis
