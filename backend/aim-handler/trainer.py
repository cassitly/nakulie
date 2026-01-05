"""
Training System for AIM Handler
Supports both real-time learning and batch training from datasets
"""

from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime
import time

from agent import CompositeAgentSystem


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    max_samples: Optional[int] = None
    save_interval: int = 100
    verbose: bool = True
    extract_concepts: bool = True
    learn_relations: bool = True


class TrainingMetrics:
    """Track training metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.samples_processed = 0
        self.concepts_learned = 0
        self.relations_learned = 0
        self.tokens_processed = 0
        self.batches_completed = 0
        self.errors = []
        
    def update(self, samples: int = 0, concepts: int = 0, relations: int = 0, tokens: int = 0):
        """Update metrics"""
        self.samples_processed += samples
        self.concepts_learned += concepts
        self.relations_learned += relations
        self.tokens_processed += tokens
        
    def complete_batch(self):
        """Mark batch as complete"""
        self.batches_completed += 1
        
    def add_error(self, error: str):
        """Record an error"""
        self.errors.append({
            'time': datetime.now().isoformat(),
            'error': error
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_seconds': elapsed,
            'samples_processed': self.samples_processed,
            'concepts_learned': self.concepts_learned,
            'relations_learned': self.relations_learned,
            'tokens_processed': self.tokens_processed,
            'batches_completed': self.batches_completed,
            'samples_per_second': self.samples_processed / elapsed if elapsed > 0 else 0,
            'errors_count': len(self.errors),
            'errors': self.errors[-10:]  # Last 10 errors
        }


class DatasetLoader:
    """Load and preprocess datasets"""
    
    def __init__(self):
        self.loaders = {
            'json': self._load_json,
            'jsonl': self._load_jsonl,
            'text': self._load_text,
            'csv': self._load_csv
        }
    
    def load(self, path: Path, format: str = 'auto') -> List[Dict[str, Any]]:
        """Load dataset from file"""
        if format == 'auto':
            format = self._detect_format(path)
        
        loader = self.loaders.get(format)
        if not loader:
            raise ValueError(f"Unsupported format: {format}")
        
        return loader(path)
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension"""
        ext = path.suffix.lower()
        if ext == '.json':
            return 'json'
        elif ext == '.jsonl':
            return 'jsonl'
        elif ext == '.txt':
            return 'text'
        elif ext == '.csv':
            return 'csv'
        else:
            return 'text'
    
    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return [{'text': str(data)}]
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        """Load plain text file"""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [{'text': p} for p in paragraphs]
    
    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Load CSV file (basic implementation)"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return data
            
            # Parse header
            header = [h.strip() for h in lines[0].split(',')]
            
            # Parse rows
            for line in lines[1:]:
                values = [v.strip() for v in line.split(',')]
                if len(values) == len(header):
                    data.append(dict(zip(header, values)))
        
        return data


class ConceptExtractor:
    """Extract concepts from text data"""
    
    def __init__(self):
        # Common patterns for concept extraction
        self.concept_indicators = [
            'is a', 'is an', 'are', 'refers to', 'means', 'defines',
            'called', 'known as', 'type of', 'kind of'
        ]
    
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text"""
        concepts = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            # Look for definition patterns
            for indicator in self.concept_indicators:
                if indicator in sentence.lower():
                    concept = self._parse_definition(sentence, indicator)
                    if concept:
                        concepts.append(concept)
        
        return concepts
    
    def _parse_definition(self, sentence: str, indicator: str) -> Optional[Dict[str, Any]]:
        """Parse a definition sentence"""
        lower_sentence = sentence.lower()
        idx = lower_sentence.find(indicator)
        
        if idx == -1:
            return None
        
        # Subject is before indicator
        subject = sentence[:idx].strip()
        # Definition is after indicator
        definition = sentence[idx + len(indicator):].strip()
        
        if len(subject) > 2 and len(definition) > 5:
            return {
                'name': subject.title(),
                'definition': definition,
                'attributes': {
                    'extracted': True,
                    'source': 'text_mining'
                }
            }
        
        return None
    
    def extract_relations(self, text: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Extract relations between known concepts"""
        relations = []
        
        # Relation indicators
        rel_patterns = {
            'used_for': ['used for', 'applied to', 'helps', 'enables'],
            'part_of': ['part of', 'component of', 'belongs to'],
            'related_to': ['related to', 'similar to', 'connected to'],
            'causes': ['causes', 'leads to', 'results in'],
            'requires': ['requires', 'needs', 'depends on']
        }
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            
            # Find concepts in sentence
            found_concepts = [c for c in concepts if c.lower() in lower_sent]
            
            if len(found_concepts) >= 2:
                # Find relation type
                for rel_type, patterns in rel_patterns.items():
                    for pattern in patterns:
                        if pattern in lower_sent:
                            relations.append({
                                'source': found_concepts[0],
                                'relation': rel_type,
                                'target': found_concepts[1],
                                'evidence': sentence
                            })
                            break
        
        return relations


class AITrainer:
    """Main training orchestrator"""
    
    def __init__(self, system: CompositeAgentSystem, config: Optional[TrainingConfig] = None):
        self.system = system
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()
        self.loader = DatasetLoader()
        self.extractor = ConceptExtractor()
    
    def train_from_file(self, path: Path, format: str = 'auto'):
        """Train from a file"""
        print(f"\nLoading dataset from {path}...")
        data = self.loader.load(path, format)
        print(f"Loaded {len(data)} samples")
        
        self.train_from_data(data)
    
    def train_from_data(self, data: List[Dict[str, Any]]):
        """Train from data list"""
        if self.config.max_samples:
            data = data[:self.config.max_samples]
        
        total = len(data)
        print(f"\nStarting training on {total} samples...")
        print(f"Batch size: {self.config.batch_size}")
        print("-" * 80)
        
        batch = []
        for i, item in enumerate(data, 1):
            batch.append(item)
            
            if len(batch) >= self.config.batch_size:
                self._process_batch(batch)
                batch = []
                
                if self.config.verbose and i % self.config.save_interval == 0:
                    self._print_progress(i, total)
                    self.system.save()
            
            # Save periodically
            if i % self.config.save_interval == 0:
                self.system.save()
        
        # Process remaining batch
        if batch:
            self._process_batch(batch)
        
        # Final save
        self.system.save()
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        self._print_summary()
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of samples"""
        concepts_before = len(self.system.ai_system.memory.concepts)
        
        for item in batch:
            try:
                text = self._extract_text(item)
                if not text:
                    continue
                
                # Learn from text
                self.system.process_task(text, task_type="learn")
                
                # Extract and learn concepts
                if self.config.extract_concepts:
                    concepts = self.extractor.extract_from_text(text)
                    for concept in concepts:
                        self.system.teach_concept(
                            name=concept['name'],
                            attributes=concept['attributes']
                        )
                
                # Extract and learn relations
                if self.config.learn_relations:
                    known_concepts = list(self.system.ai_system.memory.concepts.keys())
                    relations = self.extractor.extract_relations(text, known_concepts)
                    for rel in relations:
                        self.system.ai_system.memory.link_concepts(
                            rel['source'], rel['relation'], rel['target']
                        )
                        self.metrics.relations_learned += 1
                
                # Update metrics
                tokens = len(self.system.ai_system._tokenize(text))
                self.metrics.update(samples=1, tokens=tokens)
                
            except Exception as e:
                self.metrics.add_error(str(e))
        
        concepts_after = len(self.system.ai_system.memory.concepts)
        self.metrics.concepts_learned += (concepts_after - concepts_before)
        self.metrics.complete_batch()
    
    def _extract_text(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text from item"""
        # Try common text fields
        for field in ['text', 'content', 'body', 'message', 'description']:
            if field in item:
                return str(item[field])
        
        # Try to stringify the whole item
        if isinstance(item, str):
            return item
        
        return None
    
    def _print_progress(self, current: int, total: int):
        """Print training progress"""
        pct = (current / total) * 100
        metrics = self.metrics.get_summary()
        
        print(f"\nProgress: {current}/{total} ({pct:.1f}%)")
        print(f"  Concepts learned: {metrics['concepts_learned']}")
        print(f"  Relations learned: {metrics['relations_learned']}")
        print(f"  Tokens processed: {metrics['tokens_processed']:,}")
        print(f"  Speed: {metrics['samples_per_second']:.1f} samples/sec")
        
        # Show system state
        state = self.system.ai_system.inspect_state()
        print(f"  System: {state['total_concepts']} concepts, "
              f"{state['language_states']} language states")
    
    def _print_summary(self):
        """Print training summary"""
        metrics = self.metrics.get_summary()
        
        print("\nTraining Summary:")
        print(f"  Time elapsed: {metrics['elapsed_seconds']:.1f}s")
        print(f"  Samples processed: {metrics['samples_processed']}")
        print(f"  Concepts learned: {metrics['concepts_learned']}")
        print(f"  Relations learned: {metrics['relations_learned']}")
        print(f"  Tokens processed: {metrics['tokens_processed']:,}")
        print(f"  Batches completed: {metrics['batches_completed']}")
        print(f"  Average speed: {metrics['samples_per_second']:.1f} samples/sec")
        
        if metrics['errors_count'] > 0:
            print(f"\n  Errors encountered: {metrics['errors_count']}")
        
        # Final system state
        state = self.system.ai_system.inspect_state()
        print(f"\nFinal System State:")
        print(f"  Total concepts: {state['total_concepts']}")
        print(f"  Indexed keywords: {state['indexed_keywords']}")
        print(f"  Language states: {state['language_states']}")
        print(f"  Pattern chains: {state['pattern_chains']}")


class HuggingFaceDatasetTrainer:
    """Train from HuggingFace datasets (without requiring the library)"""
    
    def __init__(self, system: CompositeAgentSystem):
        self.system = system
        
    def create_synthetic_wikipedia_sample(self) -> List[Dict[str, Any]]:
        """Create synthetic Wikipedia-style data for testing"""
        return [
            {
                'text': 'Python is a high-level programming language. Python was created by Guido van Rossum and first released in 1991. Python is widely used for web development, data science, and artificial intelligence.'
            },
            {
                'text': 'Machine learning is a subset of artificial intelligence. Machine learning algorithms learn from data without being explicitly programmed. Machine learning is used for prediction, classification, and pattern recognition.'
            },
            {
                'text': 'Neural networks are computing systems inspired by biological neural networks. Neural networks consist of layers of interconnected nodes. Deep learning is a type of machine learning that uses neural networks with multiple layers.'
            },
            {
                'text': 'Data science is an interdisciplinary field. Data science uses scientific methods to extract knowledge from data. Data science combines statistics, programming, and domain expertise.'
            },
            {
                'text': 'Artificial intelligence is the simulation of human intelligence by machines. AI systems can perform tasks like learning, reasoning, and problem-solving. AI is used in robotics, natural language processing, and computer vision.'
            },
            {
                'text': 'Web development is the work of creating websites. Web development includes frontend and backend development. Popular web frameworks include Django, Flask, and React.'
            },
            {
                'text': 'Algorithms are step-by-step procedures for calculations. Algorithms are fundamental to computer science. Common algorithms include sorting, searching, and graph traversal.'
            },
            {
                'text': 'Databases are organized collections of data. Databases allow efficient storage and retrieval of information. SQL and NoSQL are two main types of databases.'
            },
            {
                'text': 'Cloud computing is the delivery of computing services over the internet. Cloud computing provides on-demand access to resources. Major cloud providers include AWS, Azure, and Google Cloud.'
            },
            {
                'text': 'Cybersecurity is the practice of protecting systems from digital attacks. Cybersecurity involves protecting networks, devices, and data. Common threats include malware, phishing, and ransomware.'
            }
        ]
    
    def create_synthetic_conversation_sample(self) -> List[Dict[str, Any]]:
        """Create synthetic conversation data"""
        return [
            {
                'text': 'Question: What is Python? Answer: Python is a versatile programming language known for its simplicity and readability.'
            },
            {
                'text': 'Question: How does machine learning work? Answer: Machine learning algorithms find patterns in data and use those patterns to make predictions or decisions.'
            },
            {
                'text': 'Question: What is the difference between AI and machine learning? Answer: AI is the broader concept of machines being smart, while machine learning is a specific approach to achieve AI.'
            },
            {
                'text': 'Question: Why is Python popular for data science? Answer: Python has excellent libraries like pandas, NumPy, and scikit-learn that make data analysis easier.'
            },
            {
                'text': 'Question: What are neural networks? Answer: Neural networks are computational models inspired by the human brain, consisting of interconnected nodes organized in layers.'
            }
        ]
    
    def create_synthetic_code_sample(self) -> List[Dict[str, Any]]:
        """Create synthetic code documentation"""
        return [
            {
                'text': 'The def keyword is used to define functions in Python. Functions are reusable blocks of code. Functions can accept parameters and return values.'
            },
            {
                'text': 'Lists are ordered collections in Python. Lists can contain elements of different types. Lists are created using square brackets.'
            },
            {
                'text': 'Loops allow repeating code execution. Python has for loops and while loops. For loops iterate over sequences, while while loops continue until a condition is false.'
            },
            {
                'text': 'Classes define object blueprints in Python. Classes use the class keyword. Objects are instances of classes.'
            },
            {
                'text': 'Exception handling manages errors in Python. Try-except blocks catch and handle exceptions. This prevents programs from crashing.'
            }
        ]


def create_training_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Create curated training datasets"""
    
    # Dataset 1: General Knowledge
    general_knowledge = [
        {'text': 'The solar system consists of the Sun and everything bound to it by gravity. The solar system includes eight planets, dwarf planets, moons, asteroids, and comets.'},
        {'text': 'Photosynthesis is the process plants use to convert light energy into chemical energy. Photosynthesis requires sunlight, water, and carbon dioxide.'},
        {'text': 'DNA is the molecule that carries genetic information. DNA stands for deoxyribonucleic acid. DNA is found in nearly all living organisms.'},
        {'text': 'The Renaissance was a period of cultural rebirth in Europe. The Renaissance began in Italy in the 14th century. The Renaissance saw advances in art, science, and literature.'},
        {'text': 'Climate change refers to long-term shifts in temperatures and weather patterns. Climate change is primarily caused by human activities. Climate change affects ecosystems worldwide.'},
    ]
    
    # Dataset 2: Technology & Programming
    technology = [
        {'text': 'Version control systems track changes to code. Git is the most popular version control system. Git enables collaboration among developers.'},
        {'text': 'APIs are application programming interfaces. APIs allow different software systems to communicate. REST and GraphQL are common API architectures.'},
        {'text': 'Containers package applications with their dependencies. Docker is a popular container platform. Containers enable consistent deployment across environments.'},
        {'text': 'Microservices are small, independent services that work together. Microservices architecture improves scalability and maintainability. Each microservice handles a specific function.'},
        {'text': 'Blockchain is a distributed ledger technology. Blockchain ensures data integrity through cryptography. Blockchain is used in cryptocurrencies and smart contracts.'},
    ]
    
    # Dataset 3: Science & Nature
    science = [
        {'text': 'Atoms are the basic units of matter. Atoms consist of protons, neutrons, and electrons. The nucleus contains protons and neutrons.'},
        {'text': 'Evolution is the change in species over time. Evolution occurs through natural selection. Charles Darwin developed the theory of evolution.'},
        {'text': 'Gravity is a fundamental force of nature. Gravity attracts objects with mass to each other. Newton described gravity mathematically.'},
        {'text': 'The periodic table organizes chemical elements. Elements are arranged by atomic number. The periodic table shows element properties and relationships.'},
        {'text': 'Cells are the basic units of life. Cells can be prokaryotic or eukaryotic. Eukaryotic cells have a nucleus and organelles.'},
    ]
    
    # Dataset 4: Mathematics & Logic
    mathematics = [
        {'text': 'Algebra uses symbols to represent numbers. Algebra helps solve equations and model relationships. Variables in algebra represent unknown values.'},
        {'text': 'Geometry studies shapes and spatial relationships. Geometry includes concepts like points, lines, and angles. Euclidean geometry is the most common type.'},
        {'text': 'Calculus deals with rates of change. Calculus has two main branches: differential and integral. Calculus is essential for physics and engineering.'},
        {'text': 'Statistics analyzes and interprets data. Statistics uses probability theory. Statistical methods help make data-driven decisions.'},
        {'text': 'Logic is the study of reasoning. Logic uses formal systems to evaluate arguments. Boolean logic is fundamental to computer science.'},
    ]
    
    return {
        'general': general_knowledge,
        'technology': technology,
        'science': science,
        'mathematics': mathematics
      }
