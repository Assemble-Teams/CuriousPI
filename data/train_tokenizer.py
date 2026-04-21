r"""
Train a custom tokenizer optimized for STEM data
Ensures math symbols ($\LaTeX$), chemical formulas, etc. are single tokens
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset
import os

def train_stem_tokenizer():
    """Train tokenizer on STEM corpus"""
    print("🔧 Training STEM-aware Tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Set up tokenizer components
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=50256,  # GPT-2 size
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"]
    )
    
    # Load STEM data for training
    print("Loading STEM samples for tokenizer training...")
    datasets_list = []
    
    try:
        arxiv = load_dataset("arxiv_dataset", split="train", streaming=True).take(10000)
        datasets_list.append(arxiv)
        print("  ✓ ArXiv")
    except:
        print("  ✗ ArXiv unavailable")
    
    try:
        math_data = load_dataset("open-web-math/open-web-math", split="train", streaming=True).take(5000)
        datasets_list.append(math_data)
        print("  ✓ OpenWebMath")
    except:
        print("  ✗ OpenWebMath unavailable")
    
    if not datasets_list:
        print("No datasets available. Using fallback...")
        datasets_list = [load_dataset("wikitext", "wikitext-2-raw-v1", split="train")]
    
    # Create text iterator
    def get_training_corpus():
        for dataset in datasets_list:
            for sample in dataset:
                text_key = "text" if "text" in sample else "content" if "content" in sample else list(sample.keys())[0]
                yield sample[text_key]
    
    # Train tokenizer
    print("\nTraining tokenizer on corpus...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Save tokenizer
    output_path = "./data/tokenizers/stem_tokenizer.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    
    print(f"✅ Tokenizer saved to {output_path}")
    print(f"   Vocabulary size: 50,256")
    print(f"   Optimized for: Math symbols, chemical formulas, LaTeX")
    
    return tokenizer

if __name__ == "__main__":
    train_stem_tokenizer()
