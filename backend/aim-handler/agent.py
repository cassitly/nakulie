"""
Composite Agent System - Multiple specialized agents working together
Each agent is transparent and can be inspected/modified
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from core import TransparentAI, Concept


class AgentRole(Enum):
    """Transparent agent roles"""
    LEARNER = "learner"
    REASONER = "reasoner"
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    VALIDATOR = "validator"


@dataclass
class AgentAction:
    """Transparent action representation"""
    agent_role: AgentRole
    action_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""


class BaseAgent:
    """Base class for transparent agents"""
    
    def __init__(self, role: AgentRole, ai_system: TransparentAI):
        self.role = role
        self.ai_system = ai_system
        self.action_history: List[AgentAction] = []
        
    def act(self, inputs: Dict[str, Any]) -> AgentAction:
        """Perform an action - must be implemented by subclass"""
        raise NotImplementedError
    
    def record_action(self, action: AgentAction):
        """Record action in history"""
        self.action_history.append(action)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'role': self.role.value,
            'total_actions': len(self.action_history),
            'action_types': {}
        }


class LearnerAgent(BaseAgent):
    """Agent specialized in learning from inputs"""
    
    def __init__(self, ai_system: TransparentAI):
        super().__init__(AgentRole.LEARNER, ai_system)
        self.learning_strategies = {
            'text': self._learn_from_text,
            'concept': self._learn_concept,
            'example': self._learn_from_example,
            'relation': self._learn_relation
        }
    
    def act(self, inputs: Dict[str, Any]) -> AgentAction:
        """Learn from input data"""
        learning_type = inputs.get('type', 'text')
        strategy = self.learning_strategies.get(learning_type, self._learn_from_text)
        
        action = AgentAction(
            agent_role=self.role,
            action_type=f"learn_{learning_type}",
            inputs=inputs
        )
        
        try:
            result = strategy(inputs)
            action.outputs = result
            action.confidence = 1.0
            action.reasoning = f"Successfully learned {learning_type} data"
        except Exception as e:
            action.outputs = {'error': str(e)}
            action.confidence = 0.0
            action.reasoning = f"Failed to learn: {e}"
        
        self.record_action(action)
        return action
    
    def _learn_from_text(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from text input"""
        text = inputs['data']
        context = inputs.get('context')
        
        before_concepts = len(self.ai_system.memory.concepts)
        before_states = len(self.ai_system.language_model.transitions)
        
        self.ai_system.learn_from_text(text, context)
        
        after_concepts = len(self.ai_system.memory.concepts)
        after_states = len(self.ai_system.language_model.transitions)
        
        return {
            'new_concepts': after_concepts - before_concepts,
            'new_states': after_states - before_states,
            'text_length': len(text)
        }
    
    def _learn_concept(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a new concept"""
        self.ai_system.learn_concept(
            name=inputs['name'],
            attributes=inputs.get('attributes', {}),
            examples=inputs.get('examples', []),
            relations=inputs.get('relations', {})
        )
        
        return {
            'concept_name': inputs['name'],
            'learned': True
        }
    
    def _learn_from_example(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from an example"""
        example = inputs['example']
        concept_name = inputs.get('concept')
        
        if concept_name:
            concept = self.ai_system.memory.get_concept(concept_name)
            if concept:
                concept.examples.append(example)
        
        self.ai_system.learn_from_text(example)
        
        return {'example_learned': True}
    
    def _learn_relation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a relation between concepts"""
        self.ai_system.memory.link_concepts(
            inputs['source'],
            inputs['relation'],
            inputs['target']
        )
        
        return {
            'relation': f"{inputs['source']} -{inputs['relation']}-> {inputs['target']}",
            'linked': True
        }


class ReasonerAgent(BaseAgent):
    """Agent specialized in reasoning about concepts and relations"""
    
    def __init__(self, ai_system: TransparentAI):
        super().__init__(AgentRole.REASONER, ai_system)
    
    def act(self, inputs: Dict[str, Any]) -> AgentAction:
        """Perform reasoning"""
        query = inputs.get('query', '')
        reasoning_type = inputs.get('type', 'infer')
        
        action = AgentAction(
            agent_role=self.role,
            action_type=f"reason_{reasoning_type}",
            inputs=inputs
        )
        
        if reasoning_type == 'infer':
            result = self._infer_relations(query)
        elif reasoning_type == 'chain':
            result = self._chain_reasoning(query)
        elif reasoning_type == 'compare':
            result = self._compare_concepts(inputs.get('concepts', []))
        else:
            result = self._general_reasoning(query)
        
        action.outputs = result
        action.confidence = result.get('confidence', 0.5)
        action.reasoning = result.get('reasoning', '')
        
        self.record_action(action)
        return action
    
    def _infer_relations(self, query: str) -> Dict[str, Any]:
        """Infer relations between concepts"""
        concepts = self.ai_system.memory.lookup(query)
        
        if len(concepts) < 2:
            return {
                'inferred_relations': [],
                'confidence': 0.0,
                'reasoning': 'Not enough concepts to infer relations'
            }
        
        relations = []
        for concept in concepts[:5]:
            for rel_type, targets in concept.relations.items():
                relations.append({
                    'source': concept.name,
                    'relation': rel_type,
                    'targets': targets
                })
        
        return {
            'inferred_relations': relations,
            'confidence': 0.8 if relations else 0.2,
            'reasoning': f'Found {len(relations)} existing relations'
        }
    
    def _chain_reasoning(self, query: str) -> Dict[str, Any]:
        """Perform chain reasoning through concept relations"""
        concepts = self.ai_system.memory.lookup(query)
        
        if not concepts:
            return {
                'chain': [],
                'confidence': 0.0,
                'reasoning': 'No starting concepts found'
            }
        
        # Build reasoning chain
        chain = [concepts[0].name]
        visited = {concepts[0].name}
        
        current = concepts[0]
        for _ in range(5):  # Max chain length
            if not current.relations:
                break
            
            # Get most common relation type
            rel_type = list(current.relations.keys())[0]
            targets = current.relations[rel_type]
            
            # Find unvisited target
            next_concept_name = None
            for target in targets:
                if target not in visited:
                    next_concept_name = target
                    break
            
            if not next_concept_name:
                break
            
            chain.append(f"-{rel_type}->")
            chain.append(next_concept_name)
            visited.add(next_concept_name)
            
            current = self.ai_system.memory.get_concept(next_concept_name)
            if not current:
                break
        
        return {
            'chain': chain,
            'confidence': 0.7,
            'reasoning': f'Built chain of {len(chain)} steps'
        }
    
    def _compare_concepts(self, concept_names: List[str]) -> Dict[str, Any]:
        """Compare multiple concepts"""
        concepts = [self.ai_system.memory.get_concept(name) for name in concept_names]
        concepts = [c for c in concepts if c]
        
        if len(concepts) < 2:
            return {
                'comparison': {},
                'confidence': 0.0,
                'reasoning': 'Need at least 2 concepts to compare'
            }
        
        # Find common attributes
        common_attrs = set(concepts[0].attributes.keys())
        for concept in concepts[1:]:
            common_attrs &= set(concept.attributes.keys())
        
        # Find common relations
        common_rels = set(concepts[0].relations.keys())
        for concept in concepts[1:]:
            common_rels &= set(concept.relations.keys())
        
        return {
            'comparison': {
                'common_attributes': list(common_attrs),
                'common_relations': list(common_rels),
                'unique_to': {
                    c.name: {
                        'attributes': list(set(c.attributes.keys()) - common_attrs),
                        'relations': list(set(c.relations.keys()) - common_rels)
                    }
                    for c in concepts
                }
            },
            'confidence': 0.9,
            'reasoning': f'Compared {len(concepts)} concepts'
        }
    
    def _general_reasoning(self, query: str) -> Dict[str, Any]:
        """General reasoning about a query"""
        understanding = self.ai_system.understand(query)
        
        return {
            'understanding': understanding['understanding'],
            'concepts_involved': len(understanding['relevant_concepts']),
            'confidence': 0.6,
            'reasoning': 'General understanding of query'
        }


class RetrieverAgent(BaseAgent):
    """Agent specialized in retrieving information"""
    
    def __init__(self, ai_system: TransparentAI):
        super().__init__(AgentRole.RETRIEVER, ai_system)
    
    def act(self, inputs: Dict[str, Any]) -> AgentAction:
        """Retrieve information"""
        query = inputs.get('query', '')
        retrieval_type = inputs.get('type', 'concepts')
        
        action = AgentAction(
            agent_role=self.role,
            action_type=f"retrieve_{retrieval_type}",
            inputs=inputs
        )
        
        if retrieval_type == 'concepts':
            result = self._retrieve_concepts(query, inputs.get('limit', 10))
        elif retrieval_type == 'patterns':
            result = self._retrieve_patterns(query)
        elif retrieval_type == 'similar':
            result = self._retrieve_similar(inputs.get('concept_name', query))
        else:
            result = self._retrieve_all(query)
        
        action.outputs = result
        action.confidence = 1.0 if result['items'] else 0.0
        action.reasoning = f"Retrieved {len(result['items'])} items"
        
        self.record_action(action)
        return action
    
    def _retrieve_concepts(self, query: str, limit: int) -> Dict[str, Any]:
        """Retrieve concepts matching query"""
        concepts = self.ai_system.memory.lookup(query)[:limit]
        
        return {
            'items': [c.to_dict() for c in concepts],
            'count': len(concepts)
        }
    
    def _retrieve_patterns(self, query: str) -> Dict[str, Any]:
        """Retrieve language patterns"""
        tokens = self.ai_system._tokenize(query)
        predictions = self.ai_system.language_model.predict_next(tokens[-3:], top_k=10)
        
        return {
            'items': [{'token': t, 'probability': p} for t, p in predictions],
            'count': len(predictions)
        }
    
    def _retrieve_similar(self, concept_name: str) -> Dict[str, Any]:
        """Retrieve concepts similar to given concept"""
        concept = self.ai_system.memory.get_concept(concept_name)
        
        if not concept:
            return {'items': [], 'count': 0}
        
        # Find concepts with common relations
        similar = []
        for other_name, other_concept in self.ai_system.memory.concepts.items():
            if other_name == concept_name:
                continue
            
            # Check for common relations
            common_rels = set(concept.relations.keys()) & set(other_concept.relations.keys())
            if common_rels:
                similar.append({
                    'name': other_name,
                    'common_relations': list(common_rels),
                    'similarity_score': len(common_rels)
                })
        
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'items': similar[:10],
            'count': len(similar)
        }
    
    def _retrieve_all(self, query: str) -> Dict[str, Any]:
        """Retrieve all information about query"""
        understanding = self.ai_system.understand(query)
        
        return {
            'items': [understanding],
            'count': 1
        }


class CompositeAgentSystem:
    """
    Main composite agent system that coordinates multiple specialized agents.
    This is transparent - you can see what each agent is doing.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.ai_system = TransparentAI(model_path)
        
        # Initialize specialized agents
        self.learner = LearnerAgent(self.ai_system)
        self.reasoner = ReasonerAgent(self.ai_system)
        self.retriever = RetrieverAgent(self.ai_system)
        
        self.agents = {
            AgentRole.LEARNER: self.learner,
            AgentRole.REASONER: self.reasoner,
            AgentRole.RETRIEVER: self.retriever
        }
        
        self.task_history: List[Dict[str, Any]] = []
    
    def process_task(self, task: str, task_type: str = "understand") -> Dict[str, Any]:
        """Process a task by coordinating agents"""
        task_record = {
            'task': task,
            'type': task_type,
            'actions': []
        }
        
        if task_type == "learn":
            # Use learner agent
            action = self.learner.act({
                'type': 'text',
                'data': task
            })
            task_record['actions'].append(self._action_to_dict(action))
            task_record['result'] = action.outputs
            
        elif task_type == "understand":
            # Use retriever then reasoner
            retrieval = self.retriever.act({
                'query': task,
                'type': 'concepts'
            })
            task_record['actions'].append(self._action_to_dict(retrieval))
            
            reasoning = self.reasoner.act({
                'query': task,
                'type': 'infer'
            })
            task_record['actions'].append(self._action_to_dict(reasoning))
            
            task_record['result'] = {
                'retrieved': retrieval.outputs,
                'reasoning': reasoning.outputs
            }
            
        elif task_type == "reason":
            # Use reasoner agent
            action = self.reasoner.act({
                'query': task,
                'type': 'chain'
            })
            task_record['actions'].append(self._action_to_dict(action))
            task_record['result'] = action.outputs
            
        else:
            # General processing
            understanding = self.ai_system.understand(task)
            task_record['result'] = understanding
        
        self.task_history.append(task_record)
        return task_record
    
    def _action_to_dict(self, action: AgentAction) -> Dict[str, Any]:
        """Convert action to dictionary"""
        return {
            'agent': action.agent_role.value,
            'action': action.action_type,
            'inputs': action.inputs,
            'outputs': action.outputs,
            'confidence': action.confidence,
            'reasoning': action.reasoning
        }
    
    def teach_concept(self, name: str, attributes: Dict[str, Any],
                     examples: Optional[List[str]] = None,
                     relations: Optional[Dict[str, List[str]]] = None):
        """Teach the system a new concept"""
        return self.learner.act({
            'type': 'concept',
            'name': name,
            'attributes': attributes,
            'examples': examples,
            'relations': relations
        })
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the system"""
        return self.process_task(question, task_type="understand")
    
    def save(self):
        """Save the entire system state"""
        self.ai_system.save_model()
        
        # Save agent histories
        agent_data = {
            'learner': self.learner.action_history,
            'reasoner': self.reasoner.action_history,
            'retriever': self.retriever.action_history
        }
        
        history_path = self.ai_system.model_path.with_suffix('.agents.json')
        with open(history_path, 'w') as f:
            json.dump({
                'task_history': self.task_history,
                'agent_stats': {
                    role.value: agent.get_stats()
                    for role, agent in self.agents.items()
                }
            }, f, indent=2)
    
    def load(self):
        """Load the system state"""
        self.ai_system.load_model()
    
    def inspect(self) -> Dict[str, Any]:
        """Inspect the complete system state"""
        return {
            'ai_system': self.ai_system.inspect_state(),
            'agents': {
                role.value: agent.get_stats()
                for role, agent in self.agents.items()
            },
            'total_tasks': len(self.task_history),
            'recent_tasks': self.task_history[-5:] if self.task_history else []
      }
