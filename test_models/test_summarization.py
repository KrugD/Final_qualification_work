import time
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig, get_device
from utils.models import load_summarization_model


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


def test_summarization(input_csv_path="correction_results.csv", output_csv_path="summarization_results.csv"):
    """Test text summarization model on corrected texts.
    
    Args:
        input_csv_path: Path to input CSV with corrected texts
        output_csv_path: Path for output CSV with summaries
        
    Returns:
        tuple: (DataFrame with summaries, dictionary with metrics)
    """
    start_time = time.time()
    
    print("Loading summarization model...")
    summarization_model, summarization_tokenizer = load_summarization_model()
    
    input_dataframe = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    
    speaker_texts = input_dataframe.groupby("speaker")["corrected_text"].apply(" ".join).reset_index()
    
    print(f"Summarizing texts for {len(speaker_texts)} speakers...")
    
    summaries = []
    successful_summaries = 0
    processing_times = []
    compression_ratios = []
    
    for _, row in speaker_texts.iterrows():
        speaker = row["speaker"]
        original_text = row["corrected_text"]
        
        print(f"Summarizing for {speaker}...")
        
        processed_text = original_text
        if len(original_text) > ModelConfig.MAX_SUMMARY_INPUT_LENGTH:
            processed_text = original_text[:ModelConfig.MAX_SUMMARY_INPUT_LENGTH] + "..."
        
        segment_start_time = time.time()
        summary_text, success_status = summarize_text(
            processed_text, 
            summarization_model, 
            summarization_tokenizer
        )
        segment_processing_time = time.time() - segment_start_time
        processing_times.append(segment_processing_time)
        
        if success_status:
            successful_summaries += 1
            
        compression_ratio = len(summary_text) / len(original_text) if len(original_text) > 0 else 0
        compression_ratios.append(compression_ratio)
        
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
        print(f"Processing time: {segment_processing_time:.2f} seconds")
        print("---")
    
    summary_dataframe = pd.DataFrame(summaries)
    summary_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    
    total_execution_time = time.time() - start_time
    metrics = {
        "execution_time": float(total_execution_time),
        "speakers_count": int(len(speaker_texts)),
        "successful_summaries": int(successful_summaries),
        "success_rate": float(successful_summaries / len(speaker_texts) if len(speaker_texts) > 0 else 0),
        "avg_processing_time_per_speaker": float(
            sum(processing_times) / len(processing_times) if processing_times else 0
        ),
        "avg_compression_ratio": float(
            sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
        ),
        "total_characters_processed": int(sum(row["original_text_length"] for row in summaries))
    }
    
    print(f"Summarization completed in {total_execution_time:.2f} seconds")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Average compression ratio: {metrics['avg_compression_ratio']:.2f}")
    print(f"Average time per speaker: {metrics['avg_processing_time_per_speaker']:.2f} seconds")
    
    return summary_dataframe, metrics


def create_meeting_minutes(summary_csv_path="summarization_results.csv", output_file_path="meeting_minutes.txt"):
    """Create meeting minutes from speaker summaries.
    
    Args:
        summary_csv_path: Path to CSV with speaker summaries
        output_file_path: Path for output meeting minutes file
    """
    summary_dataframe = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
    
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("ПРОТОКОЛ ВСТРЕЧИ\n")
        output_file.write("=" * 50 + "\n\n")
        
        for _, row in summary_dataframe.iterrows():
            output_file.write(f"СПИКЕР: {row['speaker']}\n")
            output_file.write(f"КЛЮЧЕВЫЕ ТЕЗИСЫ:\n")
            output_file.write(f"{row['summary']}\n")
            output_file.write("-" * 50 + "\n\n")
    
    print(f"Meeting minutes saved to {output_file_path}")


if __name__ == "__main__":
    result_dataframe, performance_metrics = test_summarization(
        "correction_results.csv", 
        "test_summarization_results.csv"
    )
    
    create_meeting_minutes("test_summarization_results.csv", "test_meeting_minutes.txt")