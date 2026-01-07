from dataclasses import dataclass
from typing import Literal

@dataclass
class Sentence:
  # The list of word pointers that make up the sentence,
  # in order from the starting word.
  words: list
  outputs: dict[int, dict[int, tuple[Literal['word', 'letter', 'sentence'], int]]]

@dataclass
class Word:
  # The list of letter pointers that make up the word,
  # in order from the starting letter.
  letters: list
  # A dict, that contains letters that is possible to
  # appear in this word, but not always.

  # The letter pointer, the dict of outputs affected by
  # this additional letter.
  pletters: dict[int, dict[int, tuple[Literal['word', 'letter', 'sentence'], int]]
  # The first int is the strength of the output, same
  # goes for pletters var.
  outputs: dict[int, tuple[Literal['word', 'letter', 'sentence'], int]]

@dataclass
class Letter:
  letter: str
  outputs: dict[int, dict]

nn_letter_ptr = [
  Letter(letter='a')
]

nn_word_ptr = [
  Word(letters=[8, 9], outputs={
    1: ('word', 2),
    2: ('word', 1)
  }, pletters={
    29: {
      1: {
        1: ('sentence', 2),
      }
    }
  }),
  
  Word(letters=[8, 5, 12, 12, 15], outputs={
    1: ('word', 1),
    2: ('word', 2)
  }, pletters={}),

  Word(letters=[8, 15, 23], outputs={}, pletters={}),
  Word(letters=[1, 18, 5], outputs={}, pletters={}),
  Word(letters=[25, 15, 21], outputs={}, pletters={}),
  Word(letters=[7, 15, 15, 4], outputs={}, pletters={})
]

nn_sentence_ptr = [
  Sentence(words=[3, 4, 5], outputs={
    1: ('word', 6)
  }),

  Sentence(words=[1, '!'], outputs={})
]

letter_vocab = {
  'a': 1,
  'b': 2,
  'c': 3,
  'd': 4,
  'e': 5,
  'f': 6,
  'g': 7,
  'h': 8,
  'i': 9,
  'j': 10,
  'k': 11,
  'l': 12,
  'm': 13,
  'n': 14,
  'o': 15,
  'p': 16,
  'q': 17,
  'r': 18,
  's': 19,
  't': 20,
  'u': 21,
  'v': 22,
  'w': 23,
  'x': 24,
  'y': 25,
  'z': 26,
  ' ': 27,
  '?': 28,
  '!': 29,
  ',': 30,
  '\'': 31,
  '"': 32
}

word_vocab = {
  'hi': 1,
  'hello': 2,
  'how': 3,
  'are': 4,
  'you': 5,
  'good': 6
}

sentence_vocab = {
  'how are you': 1,
  'hi!': 2
}
