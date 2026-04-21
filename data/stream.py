"""
CuriousPI Data Streaming Pipeline
Karpathy-style: Disk-less, memory-efficient, quality-first

Streams data on-the-fly from Hugging Face without downloading.
"""

from datasets import load_dataset, concatenate_datasets, IterableDataset
import random

class CuriousPIDataMixer:
    """Mix STEM datasets in Karpathy's "batter" recipe"""
    
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        
        # Data mixture ratios
        self.recipe = {
            "openwebmath": 0.20,      # Math & proofs
            "arxiv": 0.30,             # Scientific papers
            "pubmed": 0.25,            # Medical research
            "github": 0.10,            # Code
            "wikipedia": 0.15,         # General knowledge
        }
    
    def load_openwebmath(self, num_samples=50000):
        """Load OpenWebMath (20% of mix) - Pure math quality"""
        print("Loading OpenWebMath (streaming)...")
        try:
            data = load_dataset(
                "open-web-math/open-web-math",
                split="train",
                streaming=True
            )
            # Take first N samples
            data = data.take(num_samples)
            return data
        except Exception as e:
            print(f"OpenWebMath failed: {e}")
            return None
    
    def load_arxiv(self, num_samples=50000):
        """Load ArXiv papers (30% of mix) - Scientific knowledge"""
        print("Loading ArXiv (streaming)...")
        try:
            data = load_dataset(
                "arxiv_dataset",
                split="train",
                streaming=True
            )
            data = data.take(num_samples)
            return data
        except Exception as e:
            print(f"ArXiv failed: {e}")
            return None
    
    def load_pubmed(self, num_samples=50000):
        """Load PubMed (25% of mix) - Medical research"""
        print("Loading PubMed (streaming)...")
        try:
            data = load_dataset(
                "pubmed",
                split="train",
                streaming=True
            )
            data = data.take(num_samples)
            return data
        except Exception as e:
            print(f"PubMed failed: {e}")
            return None
    
    def load_github(self, num_samples=20000):
        """Load GitHub code (10% of mix) - Implementation examples"""
        print("Loading GitHub code (streaming)...")
        try:
            data = load_dataset(
                "codeparrot/github-code",
                split="train",
                streaming=True
            )
            data = data.take(num_samples)
            return data
        except Exception as e:
            print(f"GitHub failed: {e}")
            return None
    
    def load_wikipedia(self, num_samples=30000):
        """Load Wikipedia (15% of mix) - General knowledge"""
        print("Loading Wikipedia (streaming)...")
        try:
            data = load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",
                streaming=True
            )
            data = data.take(num_samples)
            return data
        except Exception as e:
            print(f"Wikipedia failed: {e}")
            return None
    
    def mix_datasets(self, total_samples=100000):
        """Combine all datasets in recipe proportions"""
        print(f"\n📚 CuriousPI Data Mixer: Combining {total_samples:,} samples")
        print("Recipe:", self.recipe)
        
        datasets_list = []
        dataset_names = []
        
        # Load each dataset
        if self.recipe["openwebmath"] > 0:
            num = int(total_samples * self.recipe["openwebmath"])
            data = self.load_openwebmath(num)
            if data:
                datasets_list.append(data)
                dataset_names.append(f"OpenWebMath ({num:,})")
        
        if self.recipe["arxiv"] > 0:
            num = int(total_samples * self.recipe["arxiv"])
            data = self.load_arxiv(num)
            if data:
                datasets_list.append(data)
                dataset_names.append(f"ArXiv ({num:,})")
        
        if self.recipe["pubmed"] > 0:
            num = int(total_samples * self.recipe["pubmed"])
            data = self.load_pubmed(num)
            if data:
                datasets_list.append(data)
                dataset_names.append(f"PubMed ({num:,})")
        
        if self.recipe["github"] > 0:
            num = int(total_samples * self.recipe["github"])
            data = self.load_github(num)
            if data:
                datasets_list.append(data)
                dataset_names.append(f"GitHub ({num:,})")
        
        if self.recipe["wikipedia"] > 0:
            num = int(total_samples * self.recipe["wikipedia"])
            data = self.load_wikipedia(num)
            if data:
                datasets_list.append(data)
                dataset_names.append(f"Wikipedia ({num:,})")
        
        print("\n✅ Loaded datasets:")
        for name in dataset_names:
            print(f"  - {name}")
        
        # Concatenate
        if len(datasets_list) > 1:
            combined = concatenate_datasets(datasets_list)
        else:
            combined = datasets_list[0] if datasets_list else None
        
        return combined

# Usage
if __name__ == "__main__":
    mixer = CuriousPIDataMixer()
    dataset = mixer.mix_datasets(total_samples=100000)
    
    if dataset:
        print(f"\n✅ Ready to train! Total samples: {len(dataset) if hasattr(dataset, '__len__') else 'streaming'}")
        # Show sample
        sample = next(iter(dataset))
        print(f"\nSample preview:")
        text_key = "text" if "text" in sample else "content" if "content" in sample else list(sample.keys())[0]
        print(f"Text: {sample[text_key][:200]}...")
