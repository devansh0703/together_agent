import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from PIL import Image
import pytesseract
import PyPDF2
import docx
import json
import re
from typing import Dict, List, Any, Optional, Union
import requests
from together import Together
import warnings
import os
import tempfile
warnings.filterwarnings('ignore')

# Your DataAnalystAgent class (paste your existing class here)
class DataAnalystAgent:
    def __init__(self, api_key: str):
        """
        Initialize the Data Analyst Agent with Together.ai API
        
        Args:
            api_key (str): Together.ai API key
        """
        self.client = Together(api_key=api_key)
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.data = None
        self.data_info = {}
        self.conversation_history = []
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load and process different types of documents
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            Dict containing processed data and metadata
        """
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension in ['csv']:
                return self._load_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(file_path)
            elif file_extension == 'pdf':
                return self._load_pdf(file_path)
            elif file_extension in ['doc', 'docx']:
                return self._load_word(file_path)
            elif file_extension == 'txt':
                return self._load_text(file_path)
            elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                return self._load_image(file_path)
            else:
                return {"error": f"Unsupported file format: {file_extension}"}
                
        except Exception as e:
            return {"error": f"Error loading file: {str(e)}"}
    
    def _load_csv(self, file_path: str) -> Dict[str, Any]:
        """Load CSV file"""
        try:
            df = pd.read_csv(file_path)
            self.data = df
            self.data_info = {
                'type': 'tabular',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'file_type': 'csv'
            }
            return {
                'success': True,
                'data_preview': df.head().to_dict(),
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error loading CSV: {str(e)}"}
    
    def _load_excel(self, file_path: str) -> Dict[str, Any]:
        """Load Excel file"""
        try:
            df = pd.read_excel(file_path)
            self.data = df
            self.data_info = {
                'type': 'tabular',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'file_type': 'excel'
            }
            return {
                'success': True,
                'data_preview': df.head().to_dict(),
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error loading Excel: {str(e)}"}
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            self.data = text
            self.data_info = {
                'type': 'text',
                'length': len(text),
                'pages': len(pdf_reader.pages),
                'file_type': 'pdf'
            }
            return {
                'success': True,
                'text_preview': text[:1000] + "..." if len(text) > 1000 else text,
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error loading PDF: {str(e)}"}
    
    def _load_word(self, file_path: str) -> Dict[str, Any]:
        """Extract text from Word document"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            self.data = text
            self.data_info = {
                'type': 'text',
                'length': len(text),
                'paragraphs': len(doc.paragraphs),
                'file_type': 'word'
            }
            return {
                'success': True,
                'text_preview': text[:1000] + "..." if len(text) > 1000 else text,
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error loading Word document: {str(e)}"}
    
    def _load_text(self, file_path: str) -> Dict[str, Any]:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            self.data = text
            self.data_info = {
                'type': 'text',
                'length': len(text),
                'lines': len(text.split('\n')),
                'file_type': 'text'
            }
            return {
                'success': True,
                'text_preview': text[:1000] + "..." if len(text) > 1000 else text,
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error loading text file: {str(e)}"}
    
    def _load_image(self, file_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            self.data = text
            self.data_info = {
                'type': 'image_text',
                'image_size': image.size,
                'text_length': len(text),
                'file_type': 'image'
            }
            return {
                'success': True,
                'extracted_text': text[:1000] + "..." if len(text) > 1000 else text,
                'info': self.data_info
            }
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        analysis = {}
        
        if self.data_info['type'] == 'tabular':
            analysis = self._analyze_tabular_data()
        elif self.data_info['type'] in ['text', 'image_text']:
            analysis = self._analyze_text_data()
        
        return analysis
    
    def _analyze_tabular_data(self) -> Dict[str, Any]:
        """Analyze tabular data (CSV, Excel)"""
        df = self.data
        
        analysis = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'missing_data': {
                'missing_counts': df.isnull().sum().to_dict(),
                'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'statistical_summary': {},
            'categorical_analysis': {},
            'correlations': {}
        }
        
        # Statistical summary for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['statistical_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                analysis['categorical_analysis'][col] = {
                    'unique_values': df[col].nunique(),
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            analysis['correlations'] = df[numeric_cols].corr().to_dict()
        
        return analysis
    
    def _analyze_text_data(self) -> Dict[str, Any]:
        """Analyze text data"""
        text = self.data
        
        analysis = {
            'basic_info': {
                'character_count': len(text),
                'word_count': len(text.split()),
                'line_count': len(text.split('\n')),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            },
            'word_frequency': {},
            'common_phrases': []
        }
        
        # Word frequency analysis
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Filter out short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top 20 most common words
        analysis['word_frequency'] = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return analysis
    
    def create_visualization(self, viz_type: str, **kwargs) -> Optional[str]:
        """
        Create visualizations based on the data
        
        Args:
            viz_type (str): Type of visualization
            **kwargs: Additional parameters for visualization
            
        Returns:
            Base64 encoded image or Plotly JSON
        """
        if self.data is None or self.data_info['type'] != 'tabular':
            return None
        
        df = self.data
        
        try:
            if viz_type == 'correlation_heatmap':
                return self._create_correlation_heatmap(df)
            elif viz_type == 'distribution':
                return self._create_distribution_plot(df, kwargs.get('column'))
            elif viz_type == 'scatter':
                return self._create_scatter_plot(df, kwargs.get('x'), kwargs.get('y'))
            elif viz_type == 'bar':
                return self._create_bar_plot(df, kwargs.get('column'))
            elif viz_type == 'line':
                return self._create_line_plot(df, kwargs.get('x'), kwargs.get('y'))
            elif viz_type == 'box':
                return self._create_box_plot(df, kwargs.get('column'))
            else:
                return None
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_distribution_plot(self, df: pd.DataFrame, column: str) -> str:
        """Create distribution plot"""
        if column not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        if df[column].dtype in ['int64', 'float64']:
            plt.hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {column}')
        else:
            value_counts = df[column].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.title(f'Top 10 Values in {column}')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create scatter plot"""
        if x_col not in df.columns or y_col not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_bar_plot(self, df: pd.DataFrame, column: str) -> str:
        """Create bar plot"""
        if column not in df.columns:
            return None
        
        plt.figure(figsize=(12, 6))
        value_counts = df[column].value_counts().head(15)
        plt.bar(range(len(value_counts)), value_counts.values)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Top 15 Values in {column}')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create line plot"""
        if x_col not in df.columns or y_col not in df.columns:
            return None
        
        plt.figure(figsize=(12, 6))
        df_sorted = df.sort_values(x_col)
        plt.plot(df_sorted[x_col], df_sorted[y_col], marker='o', markersize=3)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} over {x_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_box_plot(self, df: pd.DataFrame, column: str) -> str:
        """Create box plot"""
        if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
            return None
        
        plt.figure(figsize=(8, 6))
        plt.boxplot(df[column].dropna())
        plt.ylabel(column)
        plt.title(f'Box Plot of {column}')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def ask_question(self, question: str) -> str:
        """
        Answer questions about the loaded data using the LLM
        
        Args:
            question (str): User's question about the data
            
        Returns:
            str: AI-generated answer
        """
        if self.data is None:
            return "No data has been loaded. Please load a document first."
        
        # Prepare context based on data type
        context = self._prepare_context_for_llm()
        
        # Create the prompt
        prompt = self._create_prompt(question, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Provide detailed, accurate, and actionable insights based on the data provided. Use specific numbers and examples from the data when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer
            })
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _prepare_context_for_llm(self) -> str:
        """Prepare context information for the LLM"""
        context = f"Data Type: {self.data_info['type']}\n"
        context += f"File Type: {self.data_info['file_type']}\n\n"
        
        if self.data_info['type'] == 'tabular':
            df = self.data
            context += f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            context += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample data
            context += "Sample Data (first 5 rows):\n"
            context += df.head().to_string() + "\n\n"
            
            # Add statistical summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                context += "Statistical Summary:\n"
                context += df[numeric_cols].describe().to_string() + "\n\n"
            
            # Add missing values info
            missing_info = df.isnull().sum()
            if missing_info.sum() > 0:
                context += "Missing Values:\n"
                for col, missing_count in missing_info.items():
                    if missing_count > 0:
                        context += f"{col}: {missing_count} ({missing_count/len(df)*100:.1f}%)\n"
                context += "\n"
        
        elif self.data_info['type'] in ['text', 'image_text']:
            context += f"Text Length: {self.data_info['length']} characters\n"
            context += f"Sample Text:\n{self.data[:1000]}...\n\n"
        
        return context
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a comprehensive prompt for the LLM"""
        prompt = f"""
I have loaded and analyzed a dataset. Here's the information about the data:

{context}

Based on this data, please answer the following question:
{question}

Please provide a detailed analysis that includes:
1. Direct answer to the question
2. Relevant insights from the data
3. Any patterns or trends you notice
4. Recommendations or next steps if applicable
5. Specific examples using actual data values when possible

If the question requires calculations or statistical analysis, please show your reasoning and provide specific numbers from the data.
"""
        return prompt
    
    def generate_insights(self) -> str:
        """
        Generate automatic insights about the loaded data
        """
        if self.data is None:
            return "No data has been loaded."
        
        context = self._prepare_context_for_llm()
        
        prompt = f"""
I have a dataset with the following information:

{context}

As an expert data analyst, please provide comprehensive insights about this data including:

1. **Data Overview**: Key characteristics and structure
2. **Key Findings**: Most important patterns, trends, or observations
3. **Data Quality**: Assessment of missing values, outliers, or data issues
4. **Statistical Insights**: Important statistical measures and what they mean
5. **Relationships**: Correlations or relationships between variables (if applicable)
6. **Actionable Recommendations**: What actions could be taken based on this data
7. **Potential Questions**: Interesting questions that could be explored further

Please be specific and use actual values from the data in your analysis.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Provide comprehensive, insightful, and actionable analysis of datasets. Use specific numbers and examples from the data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def suggest_visualizations(self) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations based on the data
        """
        if self.data is None or self.data_info['type'] != 'tabular':
            return []
        
        df = self.data
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            suggestions.append({
                'type': 'correlation_heatmap',
                'title': 'Correlation Heatmap',
                'description': 'Shows relationships between numerical variables',
                'parameters': {}
            })
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            suggestions.append({
                'type': 'distribution',
                'title': f'Distribution of {col}',
                'description': f'Shows the distribution of values in {col}',
                'parameters': {'column': col}
            })
        
        # Bar plots for categorical columns
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                suggestions.append({
                    'type': 'bar',
                    'title': f'Value Counts for {col}',
                    'description': f'Shows the frequency of different values in {col}',
                    'parameters': {'column': col}
                })
        
        # Scatter plots for numeric pairs
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': f'{numeric_cols[1]} vs {numeric_cols[0]}',
                'description': f'Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                'parameters': {'x': numeric_cols[0], 'y': numeric_cols[1]}
            })
        
        # Box plots for numeric columns
        for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
            suggestions.append({
                'type': 'box',
                'title': f'Box Plot of {col}',
                'description': f'Shows distribution and outliers in {col}',
                'parameters': {'column': col}
            })
        
        return suggestions

# Set page config
st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .chat-message {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4285f4;
    }
    .question {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .answer {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def initialize_agent(api_key):
    """Initialize the Data Analyst Agent"""
    try:
        agent = DataAnalystAgent(api_key)
        st.session_state.agent = agent
        return True
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return False

def load_uploaded_file(uploaded_file):
    """Load uploaded file and process it"""
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Load with agent
        result = st.session_state.agent.load_document(tmp_file_path)
        
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        return result
    return None

def display_data_overview():
    """Display data overview and basic statistics"""
    if st.session_state.agent and st.session_state.agent.data is not None:
        data_info = st.session_state.agent.data_info
        
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        if data_info['type'] == 'tabular':
            df = st.session_state.agent.data
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📏 Rows", f"{df.shape[0]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📋 Columns", df.shape[1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.metric("❌ Missing Data", f"{missing_percent:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data preview
            st.subheader("📝 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            st.subheader("🔍 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        elif data_info['type'] in ['text', 'image_text']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📄 Characters", f"{data_info['length']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                word_count = len(st.session_state.agent.data.split())
                st.metric("📝 Words", f"{word_count:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                if 'pages' in data_info:
                    st.metric("📖 Pages", data_info['pages'])
                elif 'paragraphs' in data_info:
                    st.metric("📋 Paragraphs", data_info['paragraphs'])
                else:
                    st.metric("📏 Lines", data_info['lines'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Text preview
            st.subheader("📝 Text Preview")
            preview_text = st.session_state.agent.data[:2000] + "..." if len(st.session_state.agent.data) > 2000 else st.session_state.agent.data
            st.text_area("Content Preview", preview_text, height=200)

def display_visualizations():
    """Display visualization options and charts"""
    if st.session_state.agent and st.session_state.agent.data is not None and st.session_state.agent.data_info['type'] == 'tabular':
        st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
        
        df = st.session_state.agent.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Correlation Heatmap", "Distribution Plot", "Scatter Plot", "Bar Chart", "Line Plot", "Box Plot"],
            key="viz_type"
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Parameters based on visualization type
            if viz_type == "Distribution Plot":
                selected_col = st.selectbox("Select Column", all_cols, key="dist_col")
                if st.button("Generate Distribution Plot"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('distribution', column=selected_col)
                        if chart:
                            st.session_state.current_chart = chart
            
            elif viz_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
                if st.button("Generate Scatter Plot"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('scatter', x=x_col, y=y_col)
                        if chart:
                            st.session_state.current_chart = chart
            
            elif viz_type == "Bar Chart":
                selected_col = st.selectbox("Select Column", categorical_cols, key="bar_col")
                if st.button("Generate Bar Chart"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('bar', column=selected_col)
                        if chart:
                            st.session_state.current_chart = chart
            
            elif viz_type == "Line Plot":
                x_col = st.selectbox("Select X-axis", all_cols, key="line_x")
                y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
                if st.button("Generate Line Plot"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('line', x=x_col, y=y_col)
                        if chart:
                            st.session_state.current_chart = chart
            
            elif viz_type == "Box Plot":
                selected_col = st.selectbox("Select Column", numeric_cols, key="box_col")
                if st.button("Generate Box Plot"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('box', column=selected_col)
                        if chart:
                            st.session_state.current_chart = chart
            
            elif viz_type == "Correlation Heatmap":
                if st.button("Generate Correlation Heatmap"):
                    with st.spinner("Creating visualization..."):
                        chart = st.session_state.agent.create_visualization('correlation_heatmap')
                        if chart:
                            st.session_state.current_chart = chart
        
        with col2:
            # Display chart
            if 'current_chart' in st.session_state and st.session_state.current_chart:
                st.image(base64.b64decode(st.session_state.current_chart), use_column_width=True)

def display_chat_interface():
    """Display chat interface for asking questions"""
    st.markdown('<div class="section-header">💬 Ask Questions About Your Data</div>', unsafe_allow_html=True)
    
    if st.session_state.agent and st.session_state.data_loaded:
        # Question input
        question = st.text_input("Ask a question about your data:", key="question_input")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ask Question", type="primary"):
                if question:
                    with st.spinner("Analyzing your question..."):
                        answer = st.session_state.agent.ask_question(question)
                        st.session_state.conversation_history.append({
                            'question': question,
                            'answer': answer
                        })
                        # Clear the input
                        st.rerun()
        
        with col2:
            if st.button("Generate Auto Insights"):
                with st.spinner("Generating insights..."):
                    insights = st.session_state.agent.generate_insights()
                    st.session_state.conversation_history.append({
                        'question': 'Auto-generated insights',
                        'answer': insights
                    })
                    st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("💭 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}", expanded=(i==0)):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
    else:
        st.info("Please upload a dataset first to start asking questions.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Data Analyst Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("Enter Together.ai API Key:", type="password", help="Get your API key from together.ai")
        
        if api_key and not st.session_state.agent:
            if initialize_agent(api_key):
                st.success("✅ Agent initialized successfully!")
            else:
                st.error("❌ Failed to initialize agent. Please check your API key.")
        
        st.markdown("---")
        
        # File upload
        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg'],
            help="Supported formats: CSV, Excel, PDF, Word, Text, Images"
        )
        
        if uploaded_file and st.session_state.agent:
            if st.button("Load Dataset", type="primary"):
                with st.spinner("Loading and analyzing dataset..."):
                    result = load_uploaded_file(uploaded_file)
                    
                    if result and 'success' in result and result['success']:
                        st.session_state.data_loaded = True
                        st.success(f"✅ Dataset loaded successfully!")
                        st.info(f"File type: {result['info']['file_type'].upper()}")
                        st.rerun()
                    elif result and 'error' in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.error("❌ Unknown error occurred while loading the dataset.")
        
        # Sample data option
        st.markdown("---")
        st.header("🎯 Try Sample Data")
        if st.button("Load Sample Dataset") and st.session_state.agent:
            # Create sample data
            sample_data = {
                'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson', 'Frank Miller', 'Grace Lee', 'Henry Davis'],
                'Age': [25, 30, 35, 28, 32, 29, 26, 31],
                'Salary': [50000, 60000, 70000, 55000, 65000, 58000, 52000, 63000],
                'Department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT'],
                'Experience': [2, 5, 8, 3, 6, 4, 2, 7],
                'Performance_Score': [8.5, 7.2, 9.1, 8.0, 8.8, 7.5, 8.2, 9.0]
            }
            
            df = pd.DataFrame(sample_data)
            
            # Save to temporary file and load
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                result = st.session_state.agent.load_document(tmp_file.name)
                os.unlink(tmp_file.name)
            
            if result and 'success' in result:
                st.session_state.data_loaded = True
                st.success("✅ Sample dataset loaded!")
                st.rerun()
        
        # Dataset info
        if st.session_state.data_loaded:
            st.markdown("---")
            st.header("ℹ️ Dataset Info")
            data_info = st.session_state.agent.data_info
            st.write(f"**Type:** {data_info['type']}")
            st.write(f"**Format:** {data_info['file_type'].upper()}")
            if data_info['type'] == 'tabular':
                st.write(f"**Shape:** {data_info['shape']}")
                st.write(f"**Columns:** {len(data_info['columns'])}")
    
    # Main content area
    if not st.session_state.agent:
        st.info("👆 Please enter your Together.ai API key in the sidebar to get started.")
    elif not st.session_state.data_loaded:
        st.info("📁 Please upload a dataset or load sample data from the sidebar.")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "💬 AI Chat"])
        
        with tab1:
            display_data_overview()
        
        with tab2:
            display_visualizations()
        
        with tab3:
            display_chat_interface()

if __name__ == "__main__":
    main()
