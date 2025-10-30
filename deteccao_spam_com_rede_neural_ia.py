import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Carregar dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2. Converter rótulos (spam / ham → 1 / 0)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Tokenização (conversão de palavras em números)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 5. Padronizar o tamanho das sequências
maxlen = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# 6. Criar modelo de rede neural LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=maxlen),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Treinar modelo
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# 8. Avaliar
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"\n✅ Acurácia final: {acc:.2f}")

# 9. Testar com texto manual
while True:
    msg = input("\nDigite uma mensagem para testar (ou 'sair'): ")
    if msg.lower() == "sair":
        break
    seq = tokenizer.texts_to_sequences([msg])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(pad)[0][0]
    print("📩 SPAM" if pred > 0.5 else "💬 Não é SPAM")
