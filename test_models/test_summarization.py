import time
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig, get_device
from utils.models import load_summarization_model


def save_summarization_to_txt(dataframe, output_txt_path):
    """Save summarization results to text file.
    
    Args:
        dataframe: DataFrame with summarization results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("TEXT SUMMARIZATION RESULTS\n")
        file.write("=" * 50 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Original Text Length: {row['original_text_length']} chars\n")
            file.write(f"Summary Length: {row['summary_length']} chars\n")
            file.write(f"Compression Ratio: {row['compression_ratio']:.2f}\n")
            file.write(f"Summary: {row['summary']}\n")
            file.write("=" * 50 + "\n\n")


def parse_correction_from_txt(txt_file_path):
    """Parse correction results from text file.
    
    Args:
        txt_file_path: Path to correction text file
        
    Returns:
        DataFrame: Parsed correction data
    """
    data = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        current_speaker = None
        current_corrected_text = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Speaker:"):
                current_speaker = line.replace("Speaker:", "").strip()
            elif line.startswith("Corrected:"):
                current_corrected_text = line.replace("Corrected:", "").strip()
                
                # When we have speaker and corrected text, add to results
                if current_speaker and current_corrected_text:
                    data.append({
                        "speaker": current_speaker,
                        "corrected_text": current_corrected_text
                    })
                    
                    # Reset for next segment
                    current_speaker = None
                    current_corrected_text = None
    
    return pd.DataFrame(data)


def summarize_text(input_text, summarization_model, summarization_tokenizer):
    """Summarize text using the summarization model.
    
    Args:
        input_text: Text to summarize
        summarization_model: Loaded summarization model
        summarization_tokenizer: Loaded summarization tokenizer
        
    Returns:
        tuple: (summary_text, success_status)
    """
    try:
        prompt_text = f"<LM> Сократи текст.\n {input_text}"
        input_ids = torch.tensor([summarization_tokenizer.encode(prompt_text)]).to(get_device())
        
        outputs = summarization_model.generate(
            input_ids,
            eos_token_id=summarization_tokenizer.eos_token_id,
            num_beams=5,
            min_new_tokens=17,
            max_new_tokens=200,
            do_sample=True,
            no_repeat_ngram_size=4,
            top_p=0.9
        )
        
        summary = summarization_tokenizer.decode(outputs[0][1:], skip_special_tokens=True)
        return summary.strip(), True
        
    except Exception as error:
        print(f"Error summarizing text: {error}")
        return input_text[:200] + "...", False


def test_summarization(input_txt_path, output_txt_path):
    """Test text summarization model on corrected texts.
    
    Args:
        input_txt_path: Path to input text file with corrected texts
        output_txt_path: Path for output text file with summaries
        
    Returns:
        DataFrame: DataFrame with summaries
    """
    start_time = time.time()
    
    print("Loading summarization model...")
    summarization_model, summarization_tokenizer = load_summarization_model()
    
    # Parse correction data from text file
    input_dataframe = parse_correction_from_txt(input_txt_path)
    
    if input_dataframe.empty:
        print("No correction data found to summarize")
        return pd.DataFrame()
    
    # Group texts by speaker
    speaker_texts = input_dataframe.groupby("speaker")["corrected_text"].apply(" ".join).reset_index()
    
    print(f"Summarizing texts for {len(speaker_texts)} speakers...")
    
    summaries = []
    
    for _, row in speaker_texts.iterrows():
        speaker = row["speaker"]
        original_text = row["corrected_text"]
        
        print(f"Summarizing for {speaker}...")
        
        processed_text = original_text
        if len(original_text) > ModelConfig.MAX_SUMMARY_INPUT_LENGTH:
            processed_text = original_text[:ModelConfig.MAX_SUMMARY_INPUT_LENGTH] + "..."
        
        summary_text, success_status = summarize_text(
            processed_text, 
            summarization_model, 
            summarization_tokenizer
        )
        
        compression_ratio = len(summary_text) / len(original_text) if len(original_text) > 0 else 0
        
        summaries.append({
            "speaker": speaker,
            "original_text_length": len(original_text),
            "summary_length": len(summary_text),
            "compression_ratio": compression_ratio,
            "summary": summary_text
        })
        
        print(f"Original text length: {len(original_text)}")
        print(f"Summary length: {len(summary_text)}")
        print(f"Compression ratio: {compression_ratio:.2f}")
        print(f"Summary: {summary_text}")
        print("---")
    
    summary_dataframe = pd.DataFrame(summaries)
    
    # Save to text file
    save_summarization_to_txt(summary_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"Summarization completed in {total_execution_time:.2f} seconds")
    
    return summary_dataframe


def create_meeting_minutes(summary_txt_path, output_file_path):
    """Create meeting minutes from speaker summaries.
    
    Args:
        summary_txt_path: Path to text file with speaker summaries
        output_file_path: Path for output meeting minutes file
    """
    # Parse summary data
    summary_dataframe = parse_correction_from_txt(summary_txt_path)  # Reusing parser since format is similar
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("ПРОТОКОЛ ВСТРЕЧИ\n")
        file.write("=" * 50 + "\n\n")
        
        for _, row in summary_dataframe.iterrows():
            file.write(f"СПИКЕР: {row['speaker']}\n")
            file.write(f"КЛЮЧЕВЫЕ ТЕЗИСЫ:\n")
            file.write(f"{row['corrected_text']}\n")
            file.write("-" * 50 + "\n\n")
    
    print(f"Meeting minutes saved to {output_file_path}")


if __name__ == "__main__":
    result_dataframe = test_summarization("correction.txt", "test_summarization.txt")
    create_meeting_minutes("test_summarization.txt", "test_meeting_minutes.txt")