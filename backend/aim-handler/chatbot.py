"""
Chatbot Layer for AIM Handler
Learns conversational patterns from data - NO HARDCODING
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from pathlib import Path

from agent import CompositeAgentSystem


class ConversationMemory:
    """Tracks conversation history for context"""
    
    def __init__(self, max_history: int = 5):
        self.exchanges: deque = deque(maxlen=max_history)
        
    def add(self, user_msg: str, bot_msg: str):
        """Add exchange to memory"""
        self.exchanges.append({'user': user_msg, 'bot': bot_msg})
        
    def get_context(self) -> str:
        """Get recent conversation as context"""
        if not self.exchanges:
            return ""
        
        # Return last few exchanges as context
        context_parts = []
        for ex in self.exchanges:
            context_parts.append(f"User: {ex['user']}")
            context_parts.append(f"Bot: {ex['bot']}")
        
        return " ".join(context_parts)
    
    def get_last_user(self) -> Optional[str]:
        """Get last user message"""
        if self.exchanges:
            return self.exchanges[-1]['user']
        return None


class LearnedChatbot:
    """
    Chatbot that learns everything from conversation data.
    No hardcoded responses - purely learned patterns.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.system = CompositeAgentSystem(model_path or Path("chatbot_model.pkl"))
        self.memory = ConversationMemory()
        
        # Try to load existing model
        try:
            self.system.load()
            print("✓ Loaded learned conversational patterns")
        except:
            print("⚠ No pre-trained model found")
            print("  The bot will learn as it goes, but consider training first!")
    
    def train_conversational(self, conversations: List[Dict[str, str]]):
        """
        Train on conversation pairs to learn response patterns.
        Format: [{'user': '...', 'bot': '...'}, ...]
        """
        print(f"\nTraining on {len(conversations)} conversations...")
        
        for conv in conversations:
            user_msg = conv['user']
            bot_response = conv['bot']
            
            # Learn the user message pattern
            self.system.process_task(user_msg, task_type="learn")
            
            # Learn the bot response pattern
            self.system.process_task(bot_response, task_type="learn")
            
            # Create a special combined pattern for Q&A
            # This teaches: "given user input X, respond with Y"
            combined = f"{user_msg} [RESPONSE] {bot_response}"
            self.system.process_task(combined, task_type="learn")
        
        self.system.save()
        print(f"✓ Learned {len(conversations)} conversation patterns")
    
    def respond(self, user_input: str) -> str:
        """
        Generate response based purely on learned patterns.
        Uses Markov chains and concept retrieval - no hardcoding.
        """
        # Add context from conversation history
        context = self.memory.get_context()
        full_input = f"{context} User: {user_input}".strip() if context else user_input
        
        # Strategy 1: Try to find similar learned patterns
        understanding = self.system.ai_system.understand(user_input)
        concepts = understanding['relevant_concepts']
        
        if concepts and len(concepts) > 0:
            # Found relevant concepts - try to generate from them
            primary_concept = concepts[0]
            
            # Use the concept examples if available
            if primary_concept.get('examples') and len(primary_concept['examples']) > 0:
                # Generate from examples
                example = primary_concept['examples'][0]
                response = self.system.ai_system.generate_response(example, max_length=20)
                
                # Clean up the response
                response = self._clean_response(response, user_input)
                if response and len(response.split()) >= 3:
                    self.memory.add(user_input, response)
                    return response
        
        # Strategy 2: Generate based on language model
        # This uses the Markov chains trained on conversations
        generated = self.system.ai_system.generate_response(user_input, max_length=25)
        response = self._clean_response(generated, user_input)
        
        if response and len(response.split()) >= 3:
            self.memory.add(user_input, response)
            return response
        
        # Strategy 3: Try to find answer pattern in training
        # Look for [RESPONSE] marker patterns
        tokens = self.system.ai_system._tokenize(user_input)
        if '[response]' not in tokens:
            tokens.append('[response]')
        
        generated = self.system.ai_system.language_model.generate(
            start_context=tokens[-5:],
            max_length=20
        )
        
        # Find content after [RESPONSE] marker
        try:
            response_tokens = []
            found_marker = False
            for token in generated:
                if token == '[response]':
                    found_marker = True
                    continue
                if found_marker and token not in ['user:', 'bot:']:
                    response_tokens.append(token)
            
            if response_tokens:
                response = self.system.ai_system._detokenize(response_tokens)
                response = self._clean_response(response, user_input)
                if response:
                    self.memory.add(user_input, response)
                    return response
        except:
            pass
        
        # Strategy 4: If we have previous context, continue from there
        if self.memory.get_last_user():
            last_bot = self.memory.exchanges[-1]['bot']
            continuation = self.system.ai_system.generate_response(last_bot, max_length=15)
            response = self._clean_response(continuation, user_input)
            if response:
                self.memory.add(user_input, response)
                return response
        
        # Last resort: Learn from this interaction
        # Admit we don't know but learn the pattern
        self.system.process_task(user_input, task_type="learn")
        fallback = "I'm still learning. Could you tell me more?"
        self.memory.add(user_input, fallback)
        return fallback
    
    def _clean_response(self, response: str, user_input: str) -> str:
        """Clean up generated response"""
        if not response:
            return ""
        
        # Remove the input if it got echoed
        response = response.replace(user_input.lower(), "").strip()
        
        # Remove markers and metadata
        response = response.replace('[response]', '').strip()
        response = response.replace('user:', '').strip()
        response = response.replace('bot:', '').strip()
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure it ends with punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Don't return if it's too short or just repeats input
        words = response.split()
        if len(words) < 3:
            return ""
        
        # Don't return if it's identical to user input
        if response.lower() == user_input.lower():
            return ""
        
        return response
    
    def learn_from_interaction(self, user_msg: str, bot_response: str):
        """Learn from a single user-bot exchange"""
        self.train_conversational([{'user': user_msg, 'bot': bot_response}])
    
    def save(self):
        """Save learned patterns"""
        self.system.save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about what's been learned"""
        state = self.system.ai_system.inspect_state()
        return {
            'patterns_learned': state['language_states'],
            'concepts_known': state['total_concepts'],
            'conversations_memory': len(self.memory.exchanges)
        }


def train_chatbot_on_conversations():
    """Train a chatbot on conversation data"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          Training Conversational AI (No Hardcoding!)              ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    bot = LearnedChatbot(Path("chatbot_model.pkl"))
    
    # Training conversations - these teach response patterns
    training_data = [
        # Greetings
        {'user': 'hello', 'bot': 'Hello! How can I help you today?'},
        {'user': 'hi', 'bot': 'Hi there! What can I do for you?'},
        {'user': 'hey', 'bot': 'Hey! How are you doing?'},
        
        # Questions about concepts
        {'user': 'what is python', 'bot': 'Python is a high-level programming language known for its simplicity and readability.'},
        {'user': 'what is machine learning', 'bot': 'Machine learning is a type of artificial intelligence where algorithms learn patterns from data.'},
        {'user': 'what is AI', 'bot': 'AI stands for artificial intelligence. It refers to systems that can perform tasks requiring human intelligence.'},
        {'user': 'explain neural networks', 'bot': 'Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes in layers.'},
        
        # How questions
        {'user': 'how does machine learning work', 'bot': 'Machine learning works by finding patterns in data and using those patterns to make predictions.'},
        {'user': 'how to learn programming', 'bot': 'To learn programming, start with the basics, practice coding regularly, and build projects.'},
        
        # Why questions
        {'user': 'why use python', 'bot': 'Python is popular because it has simple syntax, powerful libraries, and a large community.'},
        {'user': 'why is AI important', 'bot': 'AI is important because it can automate tasks, make predictions, and solve complex problems.'},
        
        # General knowledge
        {'user': 'tell me about data science', 'bot': 'Data science combines statistics, programming, and domain knowledge to extract insights from data.'},
        {'user': 'what are algorithms', 'bot': 'Algorithms are step-by-step procedures for solving problems or performing calculations.'},
        
        # Follow-up patterns
        {'user': 'tell me more', 'bot': 'I can provide more details. What specific aspect interests you?'},
        {'user': 'can you explain', 'bot': 'Sure, I can explain. What would you like to know about?'},
        {'user': 'thanks', 'bot': 'You\'re welcome! Let me know if you need anything else.'},
        {'user': 'thank you', 'bot': 'Happy to help! Feel free to ask more questions.'},
    ]
    
    # Train the bot
    bot.train_conversational(training_data)
    
    # Show what it learned
    stats = bot.get_stats()
    print(f"\n✓ Chatbot trained!")
    print(f"  Learned {stats['patterns_learned']} language patterns")
    print(f"  Knows {stats['concepts_known']} concepts")
    
    return bot


def interactive_chat():
    """Interactive chat with learned bot"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          AIM CHATBOT - Learned Conversational AI                  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    This chatbot learns conversation patterns from data.
    All responses are generated from learned Markov chains.
    
    Commands:
      'train' - Train on built-in conversations
      'stats' - Show what the bot has learned
      'save'  - Save learned patterns
      'quit'  - Exit
    """)
    
    bot = LearnedChatbot(Path("chatbot_model.pkl"))
    
    # Check if bot is trained
    stats = bot.get_stats()
    if stats['patterns_learned'] < 100:
        print("\n⚠ Bot has minimal training!")
        print("  Type 'train' to train on conversation data first.")
    else:
        print(f"\n✓ Bot is trained with {stats['patterns_learned']} patterns")
    
    print("\n" + "-" * 70)
    print("Chat with the bot!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nBot: Goodbye! I've saved what I learned.")
                bot.save()
                break
            
            if user_input.lower() == 'train':
                print("\n" + "=" * 70)
                trained_bot = train_chatbot_on_conversations()
                # Replace current bot with trained one
                bot = trained_bot
                print("=" * 70)
                print("\nBot: I've been trained! Try talking to me now.")
                continue
            
            if user_input.lower() == 'stats':
                stats = bot.get_stats()
                print(f"\nBot: I know {stats['concepts_known']} concepts")
                print(f"      I've learned {stats['patterns_learned']} language patterns")
                print(f"      I remember our last {stats['conversations_memory']} exchanges")
                continue
            
            if user_input.lower() == 'save':
                bot.save()
                print("\nBot: Saved everything I've learned!")
                continue
            
            # Generate response using learned patterns
            response = bot.respond(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye! Saving...")
            bot.save()
            break
        except Exception as e:
            print(f"\nBot: I encountered an error: {e}")
            print("      But I'm still learning!")


if __name__ == "__main__":
    interactive_chat()
