import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import drive
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
data_path = '/content/drive/My Drive/optimized_chatbot_dataset_cleaned.jsonl'
df = pd.read_json(data_path, lines=True)

# Tokenization
print("Tokenizing dataset...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 64

def tokenize_and_vectorize(texts):
    input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True) for t in texts]
    return np.array(input_ids)

question_input_ids = tokenize_and_vectorize(df['question'].tolist())
answer_input_ids = tokenize_and_vectorize(df['answer'].tolist())

# Visualization: Tokenization
example_text = df['question'][0]
tokens = tokenizer.tokenize(example_text)
token_ids = tokenizer.encode(example_text)

plt.figure(figsize=(12, 3))
plt.bar(range(len(tokens)), [1] * len(tokens), tick_label=tokens)
plt.title(f"Tokenization: '{textwrap.shorten(example_text, width=50)}'")
plt.xticks(rotation=45)
plt.show()
print(f"Token IDs: {token_ids}")

# Visualization: Vectorization
plt.figure(figsize=(10, 6))
sns.heatmap(question_input_ids[:10], annot=False, cmap='Blues')
plt.title("Tokenized Input Sequences (First 10 Examples)")
plt.xlabel("Token Position")
plt.ylabel("Example Index")
plt.show()

# Train-Test Split
train_enc, val_enc, train_dec, val_dec = train_test_split(question_input_ids, answer_input_ids, test_size=0.2, random_state=42)

# Model Building
encoder_inputs = Input(shape=(max_seq_length,), name='encoder_inputs')
encoder_embedding = Embedding(tokenizer.vocab_size, 128)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(64, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_seq_length,), name='decoder_inputs')
decoder_embedding = Embedding(tokenizer.vocab_size, 128)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

attention = Attention(name='attention_layer')([decoder_outputs, encoder_outputs])
decoder_combined = Concatenate()([decoder_outputs, attention])
decoder_dense = Dense(tokenizer.vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training Visualization Callback
class TrainingVisualizer(Callback):
    def __init__(self):
        self.epoch = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.fig = make_subplots(rows=1, cols=2, subplot_titles=('Training/Validation Loss', 'Training/Validation Accuracy'))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch + 1)
        for k in self.history:
            self.history[k].append(logs.get(k))

        # Keep previous graphs and update them
        self.fig.add_trace(go.Scatter(x=self.epoch, y=self.history['loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.epoch, y=self.history['val_loss'], name='Validation Loss', line=dict(color='red')), row=1, col=1)

        self.fig.add_trace(go.Scatter(x=self.epoch, y=self.history['accuracy'], name='Train Accuracy', line=dict(color='blue')), row=1, col=2)
        self.fig.add_trace(go.Scatter(x=self.epoch, y=self.history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')), row=1, col=2)

        self.fig.update_layout(height=400, width=900, title_text='Training Progress')
        self.fig.show()

# Training with visualization
model_path = '/content/drive/My Drive/closedai_model.h5'
vis = TrainingVisualizer()
history = model.fit([train_enc, train_dec], train_dec, validation_data=([val_enc, val_dec], val_dec), epochs=10, batch_size=16, callbacks=[vis, EarlyStopping(monitor='val_loss', patience=3), ModelCheckpoint(model_path, save_best_only=True)])
