"""
AIM Handler - Standalone Test Application
This demonstrates the transparent AI system with comprehensive training.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent import CompositeAgentSystem
from trainer import AITrainer, TrainingConfig, HuggingFaceDatasetTrainer, create_training_datasets
import json


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent))


def train_comprehensive():
    """Comprehensive training with multiple datasets"""
    print_header("Comprehensive Training Mode")
    
    system = CompositeAgentSystem(Path("aim_model.pkl"))
    
    # Load existing model if available
    try:
        system.load()
        print("✓ Loaded existing model")
        state = system.ai_system.inspect_state()
        print(f"  Starting with: {state['total_concepts']} concepts, "
              f"{state['language_states']} language states\n")
    except:
        print("Starting with fresh model\n")
    
    # Create trainer
    config = TrainingConfig(
        batch_size=5,
        verbose=True,
        extract_concepts=True,
        learn_relations=True
    )
    trainer = AITrainer(system, config)
    
    # Get all datasets
    datasets = create_training_datasets()
    hf_trainer = HuggingFaceDatasetTrainer(system)
    
    print("Available datasets:")
    print("  1. General Knowledge (5 samples)")
    print("  2. Technology & Programming (5 samples)")
    print("  3. Science & Nature (5 samples)")
    print("  4. Mathematics & Logic (5 samples)")
    print("  5. Wikipedia-style (10 samples)")
    print("  6. Conversation Q&A (5 samples)")
    print("  7. Code Documentation (5 samples)")
    print("  8. ALL DATASETS (40 samples)")
    print()
    
    # Train on all by default
    print("Training on ALL datasets...\n")
    
    # Train on each dataset
    print_header("Phase 1: General Knowledge")
    trainer.train_from_data(datasets['general'])
    
    print_header("Phase 2: Technology & Programming")
    trainer.train_from_data(datasets['technology'])
    
    print_header("Phase 3: Science & Nature")
    trainer.train_from_data(datasets['science'])
    
    print_header("Phase 4: Mathematics & Logic")
    trainer.train_from_data(datasets['mathematics'])
    
    print_header("Phase 5: Wikipedia-style Data")
    trainer.train_from_data(hf_trainer.create_synthetic_wikipedia_sample())
    
    print_header("Phase 6: Conversation Q&A")
    trainer.train_from_data(hf_trainer.create_synthetic_conversation_sample())
    
    print_header("Phase 7: Code Documentation")
    trainer.train_from_data(hf_trainer.create_synthetic_code_sample())
    
    print_header("Training Complete!")
    
    # Show final state
    state = system.ai_system.inspect_state()
    print(f"\nFinal System State:")
    print(f"  Total concepts: {state['total_concepts']}")
    print(f"  Indexed keywords: {state['indexed_keywords']}")
    print(f"  Language states: {state['language_states']}")
    print(f"  Pattern chains: {state['pattern_chains']}")
    
    print(f"\nMost Activated Concepts:")
    for name, count in state.get('most_activated_concepts', [])[:10]:
        print(f"  {name}: {count} activations")
    
    system.save()
    print("\n✓ Model saved to 'aim_model.pkl'")


def demo_basic_learning():
    """Demonstrate basic learning capabilities"""
    print_header("Demo 1: Basic Learning with Training Data")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    
    # Train on a small dataset first
    print("Pre-training on technology dataset...")
    config = TrainingConfig(batch_size=5, verbose=False)
    trainer = AITrainer(system, config)
    
    datasets = create_training_datasets()
    trainer.train_from_data(datasets['technology'])
    
    print(f"✓ Pre-training complete!")
    state = system.ai_system.inspect_state()
    print(f"  Learned {state['total_concepts']} concepts\n")
    
    # Teach some specific concepts
    print("Teaching concept: Python")
    system.teach_concept(
        name="Python",
        attributes={
            "type": "programming language",
            "paradigm": "multi-paradigm",
            "created": "1991"
        },
        examples=[
            "Python is a high-level programming language",
            "Python uses indentation for code blocks",
            "Python has dynamic typing"
        ],
        relations={
            "used_for": ["web development", "data science", "AI"],
            "similar_to": ["Ruby", "JavaScript"]
        }
    )
    
    print("\nTeaching concept: Machine Learning")
    system.teach_concept(
        name="Machine Learning",
        attributes={
            "type": "AI technique",
            "field": "computer science"
        },
        examples=[
            "Machine learning algorithms learn from data",
            "Machine learning can make predictions",
            "Machine learning requires training data"
        ],
        relations={
            "implemented_in": ["Python", "R"],
            "types": ["supervised", "unsupervised", "reinforcement"]
        }
    )
    
    system.save()
    print("\n✓ Learning complete!")
    print(f"System now knows {len(system.ai_system.memory.concepts)} concepts")


def demo_understanding():
    """Demonstrate understanding and retrieval"""
    print_header("Demo 2: Understanding Queries")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    system.load()
    
    queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How are Python and machine learning related?",
        "What is artificial intelligence?",
        "Explain neural networks"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        result = system.query(query)
        
        if 'retrieved' in result['result']:
            concepts = result['result']['retrieved'].get('items', [])
            print(f"Found {len(concepts)} relevant concepts:")
            for concept in concepts[:3]:
                print(f"  • {concept['name']}")
                if concept.get('attributes'):
                    attrs = list(concept['attributes'].items())[:2]
                    for key, value in attrs:
                        print(f"    - {key}: {value}")


def demo_reasoning():
    """Demonstrate reasoning capabilities"""
    print_header("Demo 3: Chain Reasoning")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    system.load()
    
    print("Building reasoning chain from 'Python'...")
    result = system.process_task("Python", task_type="reason")
    
    if result['result'].get('chain'):
        chain = result['result']['chain']
        print("\nReasoning chain:")
        print("  " + " ".join(chain))
    
    print("\n\nReasoning about 'Machine Learning'...")
    result = system.process_task("Machine Learning", task_type="reason")
    
    if result['result'].get('chain'):
        chain = result['result']['chain']
        print("\nReasoning chain:")
        print("  " + " ".join(chain))


def demo_transparency():
    """Demonstrate complete transparency"""
    print_header("Demo 4: System Transparency")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    system.load()
    
    print("Complete system inspection:")
    inspection = system.inspect()
    
    print("\nAI System State:")
    for key, value in inspection['ai_system'].items():
        if key != 'most_activated_concepts':
            print(f"  {key}: {value}")
    
    print("\nMost Activated Concepts:")
    for name, count in inspection['ai_system'].get('most_activated_concepts', [])[:10]:
        print(f"  {name}: {count} activations")
    
    print("\nAgent Statistics:")
    for agent_name, stats in inspection['agents'].items():
        print(f"  {agent_name}: {stats['total_actions']} actions")
    
    print(f"\nTotal tasks processed: {inspection['total_tasks']}")


def demo_language_generation():
    """Demonstrate language generation"""
    print_header("Demo 5: Language Generation")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    system.load()
    
    prompts = [
        "Python programming",
        "Machine learning",
        "Artificial intelligence",
        "Neural networks"
    ]
    
    print("Generating text from various prompts:\n")
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        generated = system.ai_system.generate_response(prompt, max_length=20)
        print(f"Generated: {generated}\n")
    
    print("\nLanguage predictions for 'machine learning':")
    tokens = system.ai_system._tokenize("machine learning")
    predictions = system.ai_system.language_model.predict_next(tokens, top_k=5)
    for token, prob in predictions:
        print(f"  {token}: {prob:.3f}")


def interactive_mode():
    """Interactive mode for testing"""
    print_header("Interactive Mode")
    
    system = CompositeAgentSystem(Path("aim_model.pkl"))
    
    # Try to load existing model
    try:
        system.load()
        state = system.ai_system.inspect_state()
        print(f"✓ Loaded existing model")
        print(f"  Concepts: {state['total_concepts']}, States: {state['language_states']}")
    except:
        print("Starting with fresh model")
        print("Tip: Run training first with 'python entrypoint.py train'")
    
    print("\nCommands:")
    print("  learn <text>     - Learn from text")
    print("  teach <name>     - Teach a concept (simplified)")
    print("  query <question> - Query the system")
    print("  reason <topic>   - Reason about a topic")
    print("  generate <text>  - Generate continuation")
    print("  inspect          - Inspect system state")
    print("  save             - Save the model")
    print("  quit             - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input == "quit":
                break
            
            elif user_input == "inspect":
                inspection = system.inspect()
                print("\nSystem State:")
                print(f"  Concepts: {inspection['ai_system']['total_concepts']}")
                print(f"  Keywords: {inspection['ai_system']['indexed_keywords']}")
                print(f"  Language States: {inspection['ai_system']['language_states']}")
                
                if inspection['ai_system'].get('most_activated_concepts'):
                    print("\nTop Concepts:")
                    for name, count in inspection['ai_system']['most_activated_concepts'][:5]:
                        print(f"  {name}: {count}")
            
            elif user_input == "save":
                system.save()
                print("✓ Model saved")
            
            elif user_input.startswith("learn "):
                text = user_input[6:]
                result = system.process_task(text, task_type="learn")
                print(f"✓ Learned. New concepts: {result['result'].get('new_concepts', 0)}")
            
            elif user_input.startswith("teach "):
                name = user_input[6:]
                print(f"Teaching concept: {name}")
                print("Enter attributes (key=value, one per line, empty to finish):")
                attributes = {}
                while True:
                    attr = input("  ").strip()
                    if not attr:
                        break
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        attributes[key.strip()] = value.strip()
                
                system.teach_concept(name, attributes)
                print(f"✓ Taught concept: {name}")
            
            elif user_input.startswith("query "):
                question = user_input[6:]
                result = system.query(question)
                
                if 'retrieved' in result['result']:
                    concepts = result['result']['retrieved'].get('items', [])
                    print(f"\nFound {len(concepts)} concepts:")
                    for concept in concepts[:5]:
                        print(f"  • {concept['name']}")
            
            elif user_input.startswith("reason "):
                topic = user_input[7:]
                result = system.process_task(topic, task_type="reason")
                if result['result'].get('chain'):
                    print("Chain:", " ".join(result['result']['chain']))
            
            elif user_input.startswith("generate "):
                prompt = user_input[9:]
                generated = system.ai_system.generate_response(prompt, max_length=30)
                print(f"Generated: {generated}")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Save before exit
    print("\nSaving model before exit...")
    system.save()
    print("✓ Model saved. Goodbye!")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              AIM HANDLER - Transparent AI System                  ║
    ║                                                                   ║
    ║  A composite agent system using Markov chains and symbolic        ║
    ║  lookups instead of dense neural networks. The model IS the       ║
    ║  code - completely transparent and inspectable.                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            # Comprehensive training
            train_comprehensive()
        
        elif command == "demo":
            # Run all demos with trained model
            demo_basic_learning()
            demo_understanding()
            demo_reasoning()
            demo_transparency()
            demo_language_generation()
            
            print_header("Demos Complete!")
            print("Model saved to 'demo_model.pkl'")
            print("Run 'python entrypoint.py interactive' to try it yourself!")
        
        elif command == "interactive":
            interactive_mode()
        
        elif command == "chat":
            # Chatbot mode
            from chatbot import interactive_chat
            interactive_chat()
        
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python entrypoint.py train        - Train on all datasets")
            print("  python entrypoint.py demo         - Run demonstrations")
            print("  python entrypoint.py interactive  - Interactive mode")
            print("  python entrypoint.py chat         - Chatbot mode")
    
    else:
        print("Usage:")
        print("  python entrypoint.py train        - Train on comprehensive datasets")
        print("  python entrypoint.py demo         - Run demonstrations")
        print("  python entrypoint.py interactive  - Interactive mode")
        print("  python entrypoint.py chat         - Chatbot mode (learned responses)")
        print()
        print("Tip: Run 'train' first to build a knowledge base!")


if __name__ == "__main__":
    main()
