#!/usr/bin/env python3
"""
Dependency fix script for the URL fake news detector
Resolves lxml.html.clean import issues
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Fix the lxml dependency issue"""
    print("🚀 FIXING DEPENDENCIES FOR URL FAKE NEWS DETECTOR")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider activating one first.")
        print("   Command: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    
    print("\n📦 Installing dependencies...")
    
    # Try the recommended approach first
    print("\n1️⃣  Attempting to install lxml[html_clean]...")
    if run_command("pip install 'lxml[html_clean]'", "Installing lxml[html_clean]"):
        print("✅ lxml[html_clean] installed successfully!")
    else:
        print("\n2️⃣  Fallback: Installing lxml and lxml_html_clean separately...")
        
        # Install lxml first
        if run_command("pip install lxml>=4.9.0", "Installing lxml"):
            # Try to install lxml_html_clean
            if run_command("pip install lxml_html_clean", "Installing lxml_html_clean"):
                print("✅ lxml and lxml_html_clean installed successfully!")
            else:
                print("⚠️  lxml_html_clean installation failed, but lxml is installed")
        else:
            print("❌ Both approaches failed. Manual intervention required.")
            return False
    
    # Install other required packages
    print("\n3️⃣  Installing other required packages...")
    
    packages = [
        "newspaper3k>=0.2.8",
        "beautifulsoup4>=4.10.0",
        "requests>=2.25.0",
        "tldextract>=3.1.0",
        "validators>=0.18.0",
        "textblob>=0.15.3",
        "nltk>=3.6.0",
        "joblib>=1.1.0",
        "python-dotenv>=0.19.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, but continuing...")
    
    # Install spaCy model if not already installed
    print("\n4️⃣  Checking spaCy model...")
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' already installed")
        except OSError:
            print("📥 Installing spaCy model 'en_core_web_sm'...")
            if run_command("python -m spacy download en_core_web_sm", "Installing spaCy model"):
                print("✅ spaCy model installed successfully!")
            else:
                print("❌ Failed to install spaCy model")
    except ImportError:
        print("❌ spaCy not installed. Please install it first: pip install spacy")
    
    print("\n" + "=" * 60)
    print("🎯 DEPENDENCY INSTALLATION COMPLETE!")
    
    # Test if the fix worked
    print("\n🧪 Testing if the fix worked...")
    try:
        from newspaper import Article
        print("✅ newspaper3k import successful!")
        print("✅ lxml dependency issue resolved!")
        
        print("\n🎉 SUCCESS! You can now use the URL fake news detector.")
        print("\n💡 Next steps:")
        print("   1. Run 'python test_integration.py' to test the system")
        print("   2. Run 'python main.py' to use the interactive system")
        
    except ImportError as e:
        print(f"❌ Import still failing: {e}")
        print("\n🔧 Additional troubleshooting needed:")
        print("   1. Try restarting your Python environment")
        print("   2. Check if you're in the correct virtual environment")
        print("   3. Try: pip uninstall newspaper3k lxml lxml_html_clean")
        print("   4. Then: pip install newspaper3k lxml[html_clean]")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
