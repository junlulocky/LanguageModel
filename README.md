## Language Model

### Character level language model
The dataset "plaintext.txt" is a animal story called [The sun and the wind](http://www.english-for-students.com/The-Sun-and-The-Wind.html).

```python
|-charLM_RNN.py   # class of RNN for character level LM
|-mainRNNChar.py  # main test for character level RNN LM
```

### Word level language model
The dataset is from Denny Britz's tutorial.

```python
|-wordLM_RNN.py   # class of RNN for word level LM
|-wordLM_GRU.py   # class of GRU version RNN for word level LM
|-mainRNNWord.py  # main test for word level RNN LM
|-mainGRUWord.py  # main test for word level GRU LM
|-util_wordLM_GRU.py  # util for word LM GRU 
```