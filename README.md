# Complete Interactive AI Classroom

A local Streamlit teaching app covering AI, machine learning, deep learning, LLMs and generative AI through small interactive simulations and concise explanations.

## Run on Windows

Open Command Prompt in this folder, then run:

```cmd
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The browser should open automatically. If it does not, open `http://localhost:8501`.

Do **not** launch it with `py app.py`; Streamlit apps need the Streamlit runner shown above.

## Run on macOS or Linux

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Included topics

- AI: ANI, AGI, ASI, symbolic AI, expert systems, CV, robotics and NLP
- ML: supervised and unsupervised learning, reinforcement learning, training and generalization
- DL: ANN, DNN, CNN, RNN, LSTM, backpropagation and activation functions
- LLMs: tokens, embeddings, context, transformers, attention, pre-training, fine-tuning, PEFT, LoRA, RLHF, DPO, RAG, multimodality, agents, prompting and hallucination

All examples are deliberately small, local and API-free. They demonstrate mechanisms rather than production-scale performance.

## Suggested classroom sequence

1. Rules and expert systems
2. Regression/classification and features/targets
3. Clustering and Q-learning
4. Neuron, backpropagation and CNN/RNN
5. Tokenization, embeddings and attention
6. Training/alignment, RAG and agents
7. Revision quiz
