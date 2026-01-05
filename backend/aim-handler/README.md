# AIM Handler - Transparent AI Composite Agent System

A revolutionary approach to AI that **doesn't** use dense neural networks or token prediction. Instead, this system uses:

- **Markov Chains** for sequence learning
- **Symbolic Memory** with transparent concept representations
- **Composite Agents** with specialized roles
- **Code as Model** - the model literally IS the Python code

## Philosophy

Traditional AI systems are black boxes - you can't see what they've learned or how they reason. AIM Handler is **completely transparent**:

1. **Every concept is inspectable** - stored as clear Python objects
2. **Every decision is traceable** - see exactly what the system knows
3. **The model is the code** - learning updates the actual data structures
4. **No hidden weights** - everything is symbolic and readable

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Composite Agent System                    │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Learner  │  │ Reasoner │  │Retriever │         │
│  │  Agent   │  │  Agent   │  │  Agent   │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │              │                │
│       └─────────────┴──────────────┘                │
│                     │                                │
│              ┌──────▼──────┐                        │
│              │ TransparentAI│                       │
│              └──────┬──────┘                        │
│                     │                                │
│       ┌─────────────┴─────────────┐                │
│       │                           │                 │
│  ┌────▼────┐              ┌──────▼──────┐         │
│  │ Symbolic│              │   Markov    │         │
│  │ Memory  │              │   Chains    │         │
│  │         │              │             │         │
│  │Concepts │              │ Language    │         │
│  │Relations│              │ Patterns    │         │
│  └─────────┘              └─────────────┘         │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Symbolic Memory (`core.py`)

Stores concepts as transparent, inspectable objects:

```python
Concept(
    name="Python",
    attributes={"type": "programming language"},
    relations={"used_for": ["web", "AI"]},
    examples=["Python is versatile"],
    activation_count=42
)
```

### 2. Markov Chains (`core.py`)

Learns sequences without dense math:

```python
# State: ("machine", "learning")
# Next tokens: {"can": 0.4, "is": 0.3, "models": 0.3}
```

### 3. Composite Agents (`agent.py`)

Specialized agents that work together:

- **LearnerAgent**: Learns from text, concepts, and examples
- **ReasonerAgent**: Performs inference and chain reasoning
- **RetrieverAgent**: Retrieves relevant information

### 4. Transparent AI (`core.py`)

The core system that ties everything together - completely inspectable at any time.

## Usage as Library

```python
from agent import CompositeAgentSystem
from pathlib import Path

# Initialize system
system = CompositeAgentSystem(Path("my_model.pkl"))

# Teach concepts
system.teach_concept(
    name="Neural Networks",
    attributes={
        "type": "AI model",
        "architecture": "layered"
    },
    examples=[
        "Neural networks learn from data",
        "They use backpropagation for training"
    ],
    relations={
        "related_to": ["Machine Learning", "Deep Learning"]
    }
)

# Learn from text
system.process_task(
    "Machine learning is a subset of artificial intelligence",
    task_type="learn"
)

# Query the system
result = system.query("What is machine learning?")
print(result)

# Reason about concepts
reasoning = system.process_task("Python", task_type="reason")
print(reasoning)

# Inspect complete transparency
inspection = system.inspect()
print(f"Total concepts: {inspection['ai_system']['total_concepts']}")
print(f"Language states: {inspection['ai_system']['language_states']}")

# Save the model (saves the actual learned data)
system.save()

# Load existing model
system.load()
```

## Standalone Application

Run demonstrations:

```bash
python entrypoint.py demo
```

Interactive mode:

```bash
python entrypoint.py interactive
```

### Interactive Commands

- `learn <text>` - Learn from text input
- `teach <name>` - Teach a new concept interactively
- `query <question>` - Query the system
- `reason <topic>` - Perform chain reasoning
- `inspect` - Inspect complete system state
- `save` - Save the model
- `quit` - Exit

## Key Features

### 1. Complete Transparency

Every piece of learned information is stored in human-readable format:

```python
system.inspect()
# Returns:
# {
#   'ai_system': {
#     'total_concepts': 15,
#     'indexed_keywords': 47,
#     'language_states': 234,
#     'most_activated_concepts': [
#       ('Python', 12),
#       ('Machine Learning', 8)
#     ]
#   }
# }
```

### 2. Symbolic Learning

Unlike neural networks, concepts are stored symbolically:

```python
# Traditional NN: [0.234, 0.891, 0.123, ..., 0.445]  # 1000s of opaque weights
# AIM Handler: Concept("Python", attributes={...}, relations={...})  # Clear!
```

### 3. Inspectable Reasoning

See exactly how the system reasons:

```python
# Chain reasoning shows the complete path
# Python -> used_for -> Machine Learning -> requires -> Data
```

### 4. Model IS Code

The model isn't separate from the code - it's part of it. When you save the model, you're saving the actual data structures that represent what the system knows.

### 5. No Black Box Math

- **No gradient descent**
- **No backpropagation**
- **No hidden layers**
- **No dense matrix multiplication**

Just pure symbolic lookups and probability counts.

## How Learning Works

### Text Learning

1. Tokenize input text
2. Build Markov transition probabilities
3. Extract potential concepts
4. Index for fast lookup
5. Update all data structures

### Concept Learning

1. Create explicit Concept object
2. Store attributes and relations
3. Index keywords for retrieval
4. Link to related concepts
5. Build specialized pattern chains

### Understanding

1. Lookup relevant concepts from memory
2. Activate matching concepts (increment counters)
3. Predict next tokens using Markov chains
4. Find pattern matches in specialized chains
5. Synthesize understanding from activations

## Model Persistence

The model saves in two formats:

1. **Pickle** (`.pkl`) - Fast binary format for loading
2. **JSON** (`.json`) - Human-readable format for inspection

Both contain the exact same information - the complete state of what the system knows.

## Advantages

1. **Transparent** - See exactly what the system knows
2. **Debuggable** - Trace every decision
3. **Efficient** - No GPU needed, pure Python
4. **Understandable** - No complex math
5. **Modifiable** - Edit concepts directly
6. **Privacy-friendly** - Everything stays local
7. **Deterministic** - Same input = same output
8. **Educational** - Learn how AI can work

## Disadvantages

1. **Limited capacity** - Won't match GPT-4 scale
2. **Simpler reasoning** - No deep inference
3. **Manual structure** - Requires explicit teaching
4. **Storage grows** - All concepts stored explicitly

## Philosophy: Why This Approach?

Most AI systems today are:
- Opaque (can't see what they learned)
- Non-deterministic (randomness in outputs)
- Resource-heavy (need GPUs)
- Unexplainable (can't trace reasoning)

AIM Handler shows that AI can be:
- Transparent (inspect everything)
- Deterministic (reproducible)
- Lightweight (runs anywhere)
- Explainable (trace all reasoning)

This is a **proof of concept** that AI doesn't have to be a black box.

## Future Enhancements

- **Hierarchical concepts** - Concepts within concepts
- **Temporal reasoning** - Track when things were learned
- **Uncertainty quantification** - Explicit confidence scores
- **Knowledge graphs** - Visual relationship mapping
- **Self-modification** - System updates its own code
- **Distributed memory** - Share concepts across instances

## Requirements

- Python 3.8+
- No external dependencies (pure Python)

## License

Open source - use freely for any purpose.

---

*Built with transparency in mind. Because AI should be understandable.*
