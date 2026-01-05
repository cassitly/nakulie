"""
AIM Handler - Standalone Test Application
This demonstrates the transparent AI system in action.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent import CompositeAgentSystem
import json


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent))


def demo_basic_learning():
    """Demonstrate basic learning capabilities"""
    print_header("Demo 1: Basic Learning")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    
    # Teach some basic concepts
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
    
    print("\nLearning from text...")
    system.process_task(
        "Python is widely used in machine learning because it has many libraries like TensorFlow and PyTorch",
        task_type="learn"
    )
    
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
        "How are Python and machine learning related?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        result = system.query(query)
        
        if 'retrieved' in result['result']:
            concepts = result['result']['retrieved'].get('items', [])
            print(f"Found {len(concepts)} relevant concepts:")
            for concept in concepts[:3]:
                print(f"  - {concept['name']}")
                if concept.get('attributes'):
                    print(f"    Attributes: {concept['attributes']}")


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
        print(" ".join(chain))
    
    print("\n\nComparing concepts...")
    comparison = system.reasoner.act({
        'type': 'compare',
        'concepts': ['Python', 'Machine Learning']
    })
    
    if comparison.outputs.get('comparison'):
        comp = comparison.outputs['comparison']
        print("\nCommon attributes:", comp.get('common_attributes', []))
        print("Common relations:", comp.get('common_relations', []))


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
    for name, count in inspection['ai_system'].get('most_activated_concepts', [])[:5]:
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
    
    # Add more training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun and powerful.",
        "Machine learning models can predict outcomes.",
        "Data science involves analyzing large datasets.",
        "Artificial intelligence is transforming technology."
    ]
    
    print("Training language model...")
    for text in training_texts:
        system.process_task(text, task_type="learn")
    
    print("\nGenerating text from prompt: 'Python programming'")
    generated = system.ai_system.generate_response("Python programming", max_length=15)
    print(f"Generated: {generated}")
    
    print("\nLanguage predictions for 'machine learning':")
    tokens = system.ai_system._tokenize("machine learning")
    predictions = system.ai_system.language_model.predict_next(tokens, top_k=5)
    for token, prob in predictions:
        print(f"  {token}: {prob:.3f}")


def interactive_mode():
    """Interactive mode for testing"""
    print_header("Interactive Mode")
    
    system = CompositeAgentSystem(Path("demo_model.pkl"))
    
    # Try to load existing model
    try:
        system.load()
        print("Loaded existing model")
    except:
        print("Starting with fresh model")
    
    print("\nCommands:")
    print("  learn <text>     - Learn from text")
    print("  teach <name>     - Teach a concept (simplified)")
    print("  query <question> - Query the system")
    print("  reason <topic>   - Reason about a topic")
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
                print_json(system.inspect())
            
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
                print_json(result)
            
            elif user_input.startswith("reason "):
                topic = user_input[7:]
                result = system.process_task(topic, task_type="reason")
                print_json(result)
            
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
        
        if command == "demo":
            # Run all demos
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
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python entrypoint.py [demo|interactive]")
    
    else:
        print("Usage:")
        print("  python entrypoint.py demo         - Run demonstrations")
        print("  python entrypoint.py interactive  - Interactive mode")
        print()


if __name__ == "__main__":
    main()
