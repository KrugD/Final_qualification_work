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


def summarize_text(input_text, summarization_model, summarization_tokenizer):
    """Summarize text using the summarization model.
    
    Uses token-based truncation to respect model limits.
    Deterministic beam search (do_sample=False, no top_p).
    
    Args:
        input_text: Text to summarize
        summarization_model: Loaded summarization model
        summarization_tokenizer: Loaded summarization tokenizer
        
    Returns:
        tuple: (summary_text, success_status)
    """
    try:
        prompt_text = f"<LM> Создай краткое содержание текста, сохрани ключевые идеи:\n {input_text}"
        
        input_ids = summarization_tokenizer.encode(
            prompt_text,
            max_length=ModelConfig.MAX_SUMMARY_INPUT_TOKENS,
            truncation=True
        )
        input_ids = torch.tensor([input_ids]).to(get_device())
        
        outputs = summarization_model.generate(
            input_ids,
            eos_token_id=summarization_tokenizer.eos_token_id,
            num_beams=5,
            min_new_tokens=17,
            max_new_tokens=200,
            do_sample=False,
            no_repeat_ngram_size=4,
        )
        
        summary = summarization_tokenizer.decode(outputs[0][1:], skip_special_tokens=True)
        return summary.strip(), True
        
    except Exception as error:
        print(f"Error summarizing text: {error}")
        return input_text[:200] + "...", False


def perform_summarization(input_txt_path=None, output_txt_path=None, corrected_df=None,
                          summarization_model=None, summarization_tokenizer=None,
                          progress_callback=None):
    """Perform text summarization on corrected ASR results.
    
    Dynamically selects 'corrected_text' column if available, otherwise 'text'.
    
    Args:
        input_txt_path: Path to input text file with corrected results (CLI mode)
        output_txt_path: Path for output text file with summaries (CLI mode)
        corrected_df: DataFrame with corrected ASR texts (bot mode)
        summarization_model: Optional pre-loaded model (for bot mode)
        summarization_tokenizer: Optional pre-loaded tokenizer (for bot mode)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame: DataFrame with summaries
    """
    start_time = time.time()
    
    if summarization_model is None or summarization_tokenizer is None:
        print("Loading summarization model...")
        summarization_model, summarization_tokenizer = load_summarization_model()
    
    if corrected_df is not None:
        input_dataframe = corrected_df.copy()
        print("Using provided corrected DataFrame...")
    elif input_txt_path:
        from pipeline.asr import parse_asr_from_txt
        input_dataframe = parse_asr_from_txt(input_txt_path)
    else:
        print("No data provided for summarization")
        return pd.DataFrame()
    
    if input_dataframe.empty:
        print("No data found to summarize")
        return pd.DataFrame()
    
    text_col = 'corrected_text' if 'corrected_text' in input_dataframe.columns else 'text'
    
    speaker_texts = input_dataframe.groupby("speaker")[text_col].apply(" ".join).reset_index()
    speaker_texts.rename(columns={text_col: '_joined_text'}, inplace=True)
    
    print(f"Summarizing texts for {len(speaker_texts)} speakers...")
    
    summaries = []
    
    for _, row in speaker_texts.iterrows():
        speaker = row["speaker"]
        original_text = row["_joined_text"]
        
        print(f"Summarizing for {speaker}...")
        
        summary_text, success_status = summarize_text(
            original_text,
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
    
    if output_txt_path:
        save_summarization_to_txt(summary_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"Summarization completed in {total_execution_time:.2f} seconds")
    
    return summary_dataframe


if __name__ == "__main__":
    result_dataframe = perform_summarization("correction_output.txt", "summarization_output.txt")
