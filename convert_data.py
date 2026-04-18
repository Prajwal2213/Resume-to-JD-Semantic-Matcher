import json
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans 

def convert_json_to_spacy(json_file_path, output_file_path):
    nlp = spacy.blank("en")
    db = DocBin()
    
    print(f"Opening {json_file_path}...")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue 
                
            text = data.get('content', '')
            annotations = data.get('annotation', [])
            
            doc = nlp.make_doc(text)
            ents = []
            
            if annotations:
                for annot in annotations:
                    if not annot.get('label') or not annot.get('points'):
                        continue 
                        
                    start = annot['points'][0]['start']
                    end = annot['points'][0]['end']
                    label = annot['label'][0]
                    
                    # Create the span, letting spaCy snap it to valid token boundaries
                    span = doc.char_span(start, end + 1, label=label, alignment_mode="contract")
                    
                    # THE ULTIMATE FIX: 
                    # 1. Check if the span exists
                    # 2. Check if the span text has any hidden whitespaces at the edges
                    # 3. If it does, we throw it away to protect the AI
                    if span is not None:
                        if span.text.strip() == span.text:
                            ents.append(span)
                        
            # Filter out overlapping highlights
            doc.ents = filter_spans(ents)
            db.add(doc)
                
    db.to_disk(output_file_path)
    print(f"\n✅ Successfully converted and sanitized the dataset! Saved to: {output_file_path}")

if __name__ == "__main__":
    convert_json_to_spacy("kaggle_resume_dataset.json", "train_data.spacy")