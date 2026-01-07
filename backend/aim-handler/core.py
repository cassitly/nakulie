"""
AIM Handler - Transparent AI Composite Agent System
A learning system that uses Markov chains and symbolic lookups instead of dense math.
The model IS the code - learning updates the actual Python implementation.
"""

import json
import pickle
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path


@dataclass
class Concept:
    """A transparent concept representation"""
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    examples: List[str] = field(default_factory=list)
    activation_count: int = 0
    
    def activate(self):
        """Concept activation tracking"""
        self.activation_count += 1
    
    def add_relation(self, relation_type: str, target: str):
        """Add a relation to another concept"""
        if target not in self.relations[relation_type]:
            self.relations[relation_type].append(target)
    
    def to_dict(self):
        return {
            'name': self.name,
            'attributes': self.attributes,
            'relations': dict(self.relations),
            'examples': self.examples,
            'activation_count': self.activation_count
        }


class MarkovChain:
    """Transparent Markov chain for sequence learning"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions: Dict[Tuple, Counter] = defaultdict(Counter)
        self.start_states: Counter = Counter()
        
    def learn(self, sequence: List[str]):
        """Learn from a sequence"""
        if len(sequence) < self.order + 1:
            return
            
        # Track start states
        start = tuple(sequence[:self.order])
        self.start_states[start] += 1
        
        # Learn transitions
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_token = sequence[i + self.order]
            self.transitions[state][next_token] += 1
    
    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next token given context"""
        if len(context) < self.order:
            return []
            
        state = tuple(context[-self.order:])
        if state not in self.transitions:
            return []
        
        total = sum(self.transitions[state].values())
        predictions = [
            (token, count / total)
            for token, count in self.transitions[state].most_common(top_k)
        ]
        return predictions
    
    def generate(self, start_context: Optional[List[str]] = None, max_length: int = 50) -> List[str]:
        """Generate a sequence with some randomness"""
        import random
        
        if start_context is None:
            if not self.start_states:
                return []
            state = list(self.start_states.most_common(1)[0][0])
        else:
            # Pad context if too short
            if len(start_context) < self.order:
                # Try to find a start state that begins with what we have
                for start_state in self.start_states:
                    if start_state[0] == start_context[0] if start_context else True:
                        state = list(start_state)
                        break
                else:
                    # Just use most common start
                    if self.start_states:
                        state = list(self.start_states.most_common(1)[0][0])
                    else:
                        return []
            else:
                state = list(start_context[-self.order:])
        
        result = state.copy()
        
        for _ in range(max_length):
            predictions = self.predict_next(state)
            if not predictions:
                break
            
            # Sample from top predictions with some randomness
            if len(predictions) > 1 and random.random() < 0.3:
                # 30% chance to pick second best
                next_token = predictions[min(1, len(predictions)-1)][0]
            else:
                next_token = predictions[0][0]
            
            result.append(next_token)
            state = state[1:] + [next_token]
            
            # Stop at sentence boundaries
            if next_token in ['.', '?', '!']:
                break
        
        return result


class SymbolicMemory:
    """Transparent symbolic lookup memory"""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> concept names
        
    def add_concept(self, concept: Concept):
        """Add a concept to memory"""
        self.concepts[concept.name] = concept
        self._index_concept(concept)
    
    def _index_concept(self, concept: Concept):
        """Index concept for fast lookup"""
        # Index by name tokens
        for token in concept.name.lower().split():
            self.index[token].add(concept.name)
        
        # Index by attributes
        for key, value in concept.attributes.items():
            self.index[str(key).lower()].add(concept.name)
            if isinstance(value, str):
                for token in value.lower().split():
                    self.index[token].add(concept.name)
    
    def lookup(self, query: str) -> List[Concept]:
        """Lookup concepts by query"""
        tokens = query.lower().split()
        matching_names: Counter = Counter()
        
        for token in tokens:
            for concept_name in self.index.get(token, []):
                matching_names[concept_name] += 1
        
        # Return concepts sorted by relevance
        results = []
        for name, score in matching_names.most_common():
            concept = self.concepts[name]
            concept.activate()
            results.append(concept)
        
        return results
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a specific concept"""
        return self.concepts.get(name)
    
    def link_concepts(self, source_name: str, relation: str, target_name: str):
        """Create a relation between concepts"""
        if source_name in self.concepts and target_name in self.concepts:
            self.concepts[source_name].add_relation(relation, target_name)


class TransparentAI:
    """
    The main transparent AI system.
    This IS the model - all learning updates this code structure.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.memory = SymbolicMemory()
        self.language_model = MarkovChain(order=2)  # Reduced from 3 to 2 for better short-text handling
        self.pattern_chains: Dict[str, MarkovChain] = {}
        self.learning_history: List[Dict] = []
        self.model_path = model_path or Path("model_state.pkl")
        
    def learn_from_text(self, text: str, context: Optional[str] = None):
        """Learn from text input - this updates the model structure"""
        tokens = self._tokenize(text)
        
        # Learn language patterns
        self.language_model.learn(tokens)
        
        # Extract and learn concepts
        concepts = self._extract_concepts(text, tokens)
        for concept in concepts:
            self.memory.add_concept(concept)
        
        # Record learning event
        self.learning_history.append({
            'type': 'text',
            'context': context,
            'token_count': len(tokens),
            'concepts_extracted': len(concepts)
        })
    
    def learn_concept(self, name: str, attributes: Dict[str, Any], 
                     examples: Optional[List[str]] = None,
                     relations: Optional[Dict[str, List[str]]] = None):
        """Explicitly teach the system a concept"""
        concept = Concept(
            name=name,
            attributes=attributes,
            examples=examples or [],
            relations=defaultdict(list, relations or {})
        )
        self.memory.add_concept(concept)
        
        # Learn from examples
        if examples:
            for example in examples:
                tokens = self._tokenize(example)
                
                # Create specialized chain for this concept
                if name not in self.pattern_chains:
                    self.pattern_chains[name] = MarkovChain(order=2)
                self.pattern_chains[name].learn(tokens)
        
        self.learning_history.append({
            'type': 'concept',
            'name': name,
            'examples': len(examples) if examples else 0
        })
    
    def understand(self, query: str) -> Dict[str, Any]:
        """
        Understand a query by looking up concepts and patterns.
        This is transparent - we can see exactly what the system knows.
        """
        tokens = self._tokenize(query)
        
        # Lookup relevant concepts
        concepts = self.memory.lookup(query)
        
        # Get language predictions
        language_predictions = self.language_model.predict_next(tokens[-3:], top_k=5)
        
        # Find pattern matches
        pattern_matches = {}
        for concept_name, chain in self.pattern_chains.items():
            predictions = chain.predict_next(tokens[-2:], top_k=3)
            if predictions:
                pattern_matches[concept_name] = predictions
        
        return {
            'query': query,
            'tokens': tokens,
            'relevant_concepts': [c.to_dict() for c in concepts[:5]],
            'language_predictions': language_predictions,
            'pattern_matches': pattern_matches,
            'understanding': self._synthesize_understanding(concepts, pattern_matches)
        }
    
    def _synthesize_understanding(self, concepts: List[Concept], 
                                 patterns: Dict[str, List]) -> str:
        """Synthesize understanding from activated concepts and patterns"""
        if not concepts:
            return "No relevant concepts found."
        
        understanding_parts = []
        
        # Most relevant concept
        primary = concepts[0]
        understanding_parts.append(f"Primary concept: {primary.name}")
        
        if primary.attributes:
            attrs = ", ".join(f"{k}={v}" for k, v in list(primary.attributes.items())[:3])
            understanding_parts.append(f"Attributes: {attrs}")
        
        if primary.relations:
            for rel_type, targets in list(primary.relations.items())[:2]:
                understanding_parts.append(f"{rel_type}: {', '.join(targets[:3])}")
        
        # Related concepts
        if len(concepts) > 1:
            related = [c.name for c in concepts[1:4]]
            understanding_parts.append(f"Related: {', '.join(related)}")
        
        return " | ".join(understanding_parts)
    
    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """Generate a response based on learned patterns"""
        tokens = self._tokenize(prompt)
        
        # Use language model to generate
        generated_tokens = self.language_model.generate(tokens, max_length=max_length)
        
        return self._detokenize(generated_tokens)
    
    def save_model(self):
        """Save the model state - this is the actual 'model weights'"""
        # Convert transitions to serializable format
        language_transitions_serializable = {}
        for state_tuple, counter in self.language_model.transitions.items():
            # Convert tuple key to string
            key = "|".join(state_tuple)
            language_transitions_serializable[key] = dict(counter)
        
        language_starts_serializable = {}
        for state_tuple, count in self.language_model.start_states.items():
            key = "|".join(state_tuple)
            language_starts_serializable[key] = count
        
        # Convert pattern chains
        pattern_chains_serializable = {}
        for name, chain in self.pattern_chains.items():
            transitions_ser = {}
            for state_tuple, counter in chain.transitions.items():
                key = "|".join(state_tuple)
                transitions_ser[key] = dict(counter)
            
            starts_ser = {}
            for state_tuple, count in chain.start_states.items():
                key = "|".join(state_tuple)
                starts_ser[key] = count
            
            pattern_chains_serializable[name] = {
                'transitions': transitions_ser,
                'starts': starts_ser
            }
        
        state = {
            'memory_concepts': {name: c.to_dict() for name, c in self.memory.concepts.items()},
            'memory_index': {k: list(v) for k, v in self.memory.index.items()},
            'language_transitions': language_transitions_serializable,
            'language_starts': language_starts_serializable,
            'pattern_chains': pattern_chains_serializable,
            'learning_history': self.learning_history
        }
        
        # Save as pickle (with original format for efficiency)
        pickle_state = {
            'memory_concepts': {name: c.to_dict() for name, c in self.memory.concepts.items()},
            'memory_index': {k: list(v) for k, v in self.memory.index.items()},
            'language_transitions': dict(self.language_model.transitions),
            'language_starts': dict(self.language_model.start_states),
            'pattern_chains': {
                name: {
                    'transitions': dict(chain.transitions),
                    'starts': dict(chain.start_states)
                }
                for name, chain in self.pattern_chains.items()
            },
            'learning_history': self.learning_history
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(pickle_state, f)
        
        # Save as JSON for transparency (with serialized format)
        json_path = self.model_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_model(self):
        """Load the model state"""
        if not self.model_path.exists():
            return
        
        with open(self.model_path, 'rb') as f:
            state = pickle.load(f)
        
        # Restore memory
        self.memory = SymbolicMemory()
        for name, concept_dict in state['memory_concepts'].items():
            concept = Concept(
                name=concept_dict['name'],
                attributes=concept_dict['attributes'],
                relations=defaultdict(list, concept_dict['relations']),
                examples=concept_dict['examples'],
                activation_count=concept_dict['activation_count']
            )
            self.memory.concepts[name] = concept
        
        for key, names in state['memory_index'].items():
            self.memory.index[key] = set(names)
        
        # Restore language model - handle both formats
        self.language_model.transitions = defaultdict(Counter)
        transitions_data = state.get('language_transitions', {})
        
        if transitions_data:
            # Check if it's the serialized format (string keys) or pickle format (tuple keys)
            first_key = next(iter(transitions_data.keys())) if transitions_data else None
            
            if isinstance(first_key, str):
                # Serialized format - convert back to tuples
                for key_str, counter_dict in transitions_data.items():
                    key_tuple = tuple(key_str.split("|"))
                    self.language_model.transitions[key_tuple] = Counter(counter_dict)
            else:
                # Pickle format - direct load
                for key_tuple, counter_dict in transitions_data.items():
                    self.language_model.transitions[key_tuple] = Counter(counter_dict)
        
        # Restore start states
        self.language_model.start_states = Counter()
        starts_data = state.get('language_starts', {})
        
        if starts_data:
            first_key = next(iter(starts_data.keys())) if starts_data else None
            
            if isinstance(first_key, str):
                # Serialized format
                for key_str, count in starts_data.items():
                    key_tuple = tuple(key_str.split("|"))
                    self.language_model.start_states[key_tuple] = count
            else:
                # Pickle format
                self.language_model.start_states = Counter(starts_data)
        
        # Restore pattern chains
        self.pattern_chains = {}
        for name, chain_data in state.get('pattern_chains', {}).items():
            chain = MarkovChain(order=2)
            
            # Restore transitions
            transitions_data = chain_data.get('transitions', {})
            if transitions_data:
                first_key = next(iter(transitions_data.keys())) if transitions_data else None
                
                if isinstance(first_key, str):
                    # Serialized format
                    for key_str, counter_dict in transitions_data.items():
                        key_tuple = tuple(key_str.split("|"))
                        chain.transitions[key_tuple] = Counter(counter_dict)
                else:
                    # Pickle format
                    for key_tuple, counter_dict in transitions_data.items():
                        chain.transitions[key_tuple] = Counter(counter_dict)
            
            # Restore start states
            starts_data = chain_data.get('starts', {})
            if starts_data:
                first_key = next(iter(starts_data.keys())) if starts_data else None
                
                if isinstance(first_key, str):
                    # Serialized format
                    for key_str, count in starts_data.items():
                        key_tuple = tuple(key_str.split("|"))
                        chain.start_states[key_tuple] = count
                else:
                    # Pickle format
                    chain.start_states = Counter(starts_data)
            
            self.pattern_chains[name] = chain
        
        self.learning_history = state.get('learning_history', [])
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Simple detokenization"""
        text = ' '.join(tokens)
        text = text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?')
        return text
    
    def _extract_concepts(self, text: str, tokens: List[str]) -> List[Concept]:
        """Extract potential concepts from text"""
        concepts = []
        
        # Simple noun phrase extraction (very basic)
        # In a real system, this would be more sophisticated
        sentences = text.split('.')
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 2:
                # Create concept from key phrases
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) > 5:  # Minimum length
                        concept = Concept(
                            name=phrase,
                            attributes={'source': 'extracted', 'sentence': sentence.strip()},
                            examples=[sentence.strip()]
                        )
                        concepts.append(concept)
        
        return concepts
    
    def inspect_state(self) -> Dict[str, Any]:
        """Inspect the current state of the model - complete transparency"""
        return {
            'total_concepts': len(self.memory.concepts),
            'indexed_keywords': len(self.memory.index),
            'language_states': len(self.language_model.transitions),
            'pattern_chains': len(self.pattern_chains),
            'learning_events': len(self.learning_history),
            'most_activated_concepts': [
                (name, c.activation_count)
                for name, c in sorted(
                    self.memory.concepts.items(),
                    key=lambda x: x[1].activation_count,
                    reverse=True
                )[:10]
            ]
        }
