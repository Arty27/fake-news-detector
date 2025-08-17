import { AppBar, Toolbar, Typography, Box, Chip } from '@mui/material'
import { Shield, Bug, TrendingUp } from 'lucide-react'

const Header = () => {
  return (
    <AppBar 
      position="static" 
      elevation={0}
      sx={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Shield size={32} color="white" />
          <Typography 
            variant="h5" 
            component="h1" 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(45deg, #fff 30%, #f0f0f0 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            Fake News Detector
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={<TrendingUp size={16} />}
            label="AI-Powered"
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              fontWeight: 600,
              '& .MuiChip-icon': { color: 'white' }
            }}
          />
          <Chip
            icon={<Bug size={16} />}
            label="Real-time"
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              fontWeight: 600,
              '& .MuiChip-icon': { color: 'white' }
            }}
          />
        </Box>
      </Toolbar>
    </AppBar>
  )
}

export default Header
