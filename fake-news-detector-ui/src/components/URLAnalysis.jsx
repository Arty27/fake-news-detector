import { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Paper,
  Chip,
  Link
} from '@mui/material'
import { Globe, ExternalLink, AlertCircle } from 'lucide-react'
import axios from 'axios'

const URLAnalysis = ({ onAnalysisComplete, onAnalysisStart, onAnalysisEnd, isLoading }) => {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  const [isValidUrl, setIsValidUrl] = useState(false)

  const validateURL = (urlString) => {
    try {
      const urlObj = new URL(urlString)
      return urlObj.protocol === 'http:' || urlObj.protocol === 'https:'
    } catch {
      return false
    }
  }

  const handleUrlChange = (event) => {
    const newUrl = event.target.value
    setUrl(newUrl)
    setError('')
    
    if (newUrl.trim()) {
      setIsValidUrl(validateURL(newUrl.trim()))
    } else {
      setIsValidUrl(false)
    }
  }

  const handleAnalyze = async () => {
    if (!url.trim()) {
      setError('Please enter a URL to analyze')
      return
    }

    if (!isValidUrl) {
      setError('Please enter a valid HTTP or HTTPS URL')
      return
    }

    try {
      onAnalysisStart()
      setError('')

      const response = await axios.post('http://localhost:8000/url/analyze', {
        url: url.trim()
      })

      if (response.data.success) {
        // Pass both the analysis results and the original input URL
        onAnalysisComplete({
          ...response.data.data,
          original_input: url.trim(),
          input_type: 'url'
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

  const getUrlStatusColor = () => {
    if (!url.trim()) return 'default'
    return isValidUrl ? 'success' : 'error'
  }

  const getUrlStatusLabel = () => {
    if (!url.trim()) return 'No URL'
    return isValidUrl ? 'Valid URL' : 'Invalid URL'
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        <Globe size={24} style={{ marginRight: 8, verticalAlign: 'middle' }} />
        URL Article Analysis
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
          Enter the URL of an article to analyze. The system will automatically extract and analyze the content.
        </Typography>
        
        <TextField
          fullWidth
          variant="outlined"
          placeholder="https://example.com/article"
          value={url}
          onChange={handleUrlChange}
          disabled={isLoading}
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': {
              borderRadius: 2,
              fontSize: '1rem'
            }
          }}
        />

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={getUrlStatusLabel()}
              color={getUrlStatusColor()}
              size="small"
              variant="outlined"
            />
            {isValidUrl && url.trim() && (
              <Link 
                href={url.trim()} 
                target="_blank" 
                rel="noopener noreferrer"
                sx={{ display: 'flex', alignItems: 'center', gap: 0.5, textDecoration: 'none' }}
              >
                <ExternalLink size={16} />
                <Typography variant="body2" color="primary.main">
                  Open Link
                </Typography>
              </Link>
            )}
          </Box>
          
          <Button
            variant="contained"
            size="large"
            onClick={handleAnalyze}
            disabled={isLoading || !isValidUrl}
            startIcon={isLoading ? <CircularProgress size={20} /> : <Globe />}
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
            {isLoading ? 'Analyzing...' : 'Analyze URL'}
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
          💡 URL Analysis Tips
        </Typography>
        <Box component="ul" sx={{ pl: 2, mt: 1 }}>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Works best with news articles, blog posts, and factual content
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            The system automatically extracts article content, title, and metadata
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Supports most major news websites and content platforms
          </Typography>
          <Typography component="li" variant="body2" sx={{ mb: 1 }}>
            Analysis includes domain trustworthiness and live news verification
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}

export default URLAnalysis
