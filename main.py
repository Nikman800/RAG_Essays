import os
import openai
from docx import Document

class WritingStyleGPT:
    def __init__(self, user_essays_dir):
        self.user_essays_dir = user_essays_dir
        self.essay_texts = []
        self.style_analysis = ""
        # Read API key from file
        with open('OpenCV_api_key', 'r') as key_file:
            api_key = key_file.read().strip()
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_essays(self):
        """Load and analyze essays for style patterns"""
        for filename in os.listdir(self.user_essays_dir):
            if filename.startswith('~$'):
                continue
                
            file_path = os.path.join(self.user_essays_dir, filename)
            try:
                if filename.endswith('.docx'):
                    doc = Document(file_path)
                    essay_text = '\n'.join([p.text for p in doc.paragraphs])
                    self.essay_texts.append(essay_text)
                    print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        # Analyze writing style using GPT
        self.analyze_style()
        return len(self.essay_texts)
    
    def analyze_style(self):
        """Use GPT to analyze writing style patterns"""
        sample_text = "\n\n".join(self.essay_texts[:3])  # Use first 3 essays
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a writing style analyst."},
                {"role": "user", "content": f"Analyze the writing style patterns in these essays: {sample_text}"}
            ]
        )
        self.style_analysis = response.choices[0].message.content
    
    def style_transfer(self, source_text):
        """Transfer user's writing style to new text"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a style transfer expert."},
                {"role": "user", "content": f"""
                Writing style analysis: {self.style_analysis}
                
                Rewrite the following text to match this writing style:
                {source_text}
                """}
            ]
        )
        return {
            'stylized_text': response.choices[0].message.content,
            'style_analysis': self.style_analysis
        }

def read_source_text(filename):
    """Read source text from a file (.txt, .md, or .docx)"""
    if not filename.endswith(('.txt', '.md', '.docx')):
        raise ValueError("Source file must be .txt, .md, or .docx format")
        
    if filename.endswith('.docx'):
        doc = Document(filename)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()

def save_stylized_text(result, output_filename='stylized_output.docx'):
    """Save stylized text to a Word document"""
    doc = Document()
    doc.add_heading('Stylized Text', 0)
    doc.add_paragraph(result['stylized_text'])
    
    # Add style metrics
    doc.add_heading('Style Metrics', level=1)
    doc.add_paragraph(f"Style Analysis: {result['style_analysis']}")
    
    doc.save(output_filename)

def main():
    rag_styler = WritingStyleGPT('training_essays')
    num_essays = rag_styler.load_essays()
    print(f"Loaded {num_essays} essays for style learning")
    
    source_text = read_source_text('source.docx')
    result = rag_styler.style_transfer(source_text)
    
    # Save to Word document
    save_stylized_text(result)
    print(f"Stylized text saved to 'stylized_output.docx'")

if __name__ == '__main__':
    main()

# Additional Considerations:
# 1. Fine-tuning the model on user's specific writing style
# 2. Handling different essay types and genres
# 3. Implementing more sophisticated style embedding techniques
# 4. Adding privacy and data protection mechanisms