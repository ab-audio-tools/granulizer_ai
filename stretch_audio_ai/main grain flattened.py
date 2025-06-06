# Cell 1
""" !pip install nbformat tensorflow plotly ipywidgets pyqt5 scipy datasets matplotlib """

# Cell 2
import os
from glob import glob
import random
from IPython.display import Audio, display

import numpy as np
import essentia
import essentia.standard as es

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import ipywidgets as widgets

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Softmax, Input, InputLayer
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from tensorflow.keras.metrics import Mean, MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import scipy
from scipy.signal import resample

from datasets import load_dataset
import shutil


# Cell 3
print("Essentia version:", essentia.__version__)
print("Numpy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("Plotly version:", plotly.__version__)
print("IPyWidgets version:", widgets.__version__)
print("SciPy version:", scipy.__version__)

# Cell 4
original_dir = "dataset/originali"
output_dir = "dataset/granulati"
sr = 48000
window = 1024
hop=512
default_grain_size=1024
default_overlap=512

# Cell 5
# Create dataset directories if they don't exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"Directory created/verified:\n- {original_dir}\n- {output_dir}")

# Cell 6
import shutil

# Remove all files in output_dir if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' has been cleared and recreated")
else:
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' has been created")

# Cell 7


# Cell 8


# Cell 9
def clone_audiomnist():
    if not os.path.exists('audiomnist'):
        print("Cloning AudioMNIST repository...")
        os.system('git clone https://github.com/soerenab/AudioMNIST.git audiomnist')
        print("La repo è stata clonata")
    else:
        print("AudioMNIST esiste già.")

clone_audiomnist()

# Cell 10
""" def sposta_file_audiomnist():
    # Crea la directory di destinazione se non esiste
    os.makedirs(original_dir, exist_ok=True)
    
    # Path alla directory dei dati AudioMNIST
    audiomnist_dir = 'audiomnist/audioMNIST/data'
    
    # Lista per raccogliere tutti i file wav
    tutti_i_file = []
    
    # Attraversa tutte le sottocartelle e raccogli i file
    for subdir in os.listdir(audiomnist_dir):
        subdir_path = os.path.join(audiomnist_dir, subdir)
        
        if os.path.isdir(subdir_path):
            wav_files = glob(os.path.join(subdir_path, '*.wav'))
            tutti_i_file.extend(wav_files)
    
    # Rinomina e sposta i file
    for idx, wav_file in enumerate(tutti_i_file, start=1):
        nuovo_nome = f"{idx}.wav"
        destination = os.path.join(original_dir, nuovo_nome)
        
        # Copia il file con il nuovo nome
        shutil.copy2(wav_file, destination)
        print(f"File {wav_file} spostato come {destination}")
        # Rimuovi il file originale
        os.remove(wav_file)
        print(f"File {wav_file} rimosso dopo lo spostamento.")
    
    print(f"\n Spostati {len(tutti_i_file)} file audio in {original_dir}")

# Esegui la funzione
sposta_file_audiomnist() """

# Cell 11
FILE_NUMBER = 50  # Numero di file da creare

def rimuovi_file_extra():
    # Ensure the directory exists
    if not os.path.exists(original_dir):
        print(f"Directory {original_dir} non esiste.")
        return
    
    # Get list of wav files
    files = glob(os.path.join(original_dir, '*.wav'))
    
    # Process each file
    for file_path in files:
        # Extract the number from filename (assuming format like "301.wav")
        try:
            file_number = int(os.path.basename(file_path).split('.')[0])
            if file_number > FILE_NUMBER:
                os.remove(file_path)
                print(f"Rimosso file: {file_path}")
        except ValueError:
            print(f"Saltato file {file_path}: nome file non valido")
    
    # Count remaining files
    remaining_files = len(glob(os.path.join(original_dir, '*.wav')))
    print(f"\nOperazione completata. Rimangono {remaining_files} file in {original_dir}")

# Execute the function
rimuovi_file_extra()

# Cell 12


# Cell 13
loader = es.AudioLoader(filename='dataset/originali/1.wav')
audio, sr, n_channel, md5, bitrate, codec = loader()
audio_T = audio.T[0]
audio = audio_T
print(audio, sr, n_channel, md5, bitrate, codec)




# Cell 14
def carica_audio_mono(cartella, max_files=FILE_NUMBER, sr=sr):

    filepaths = glob(os.path.join(cartella, '*.wav'))[:max_files]
    audio_list = []
    for fp in filepaths:
        loader = es.AudioLoader(filename=fp)
        audio, sr, n_channel, md5, bitrate, codec = loader()
        audio_T = audio.T[0]
        audio = audio_T
        audio_list.append((fp, audio))
        # print(f"Caricato file: {fp}, durata: {len(audio)/sr:.2f} secondi")
    
    print(f"Caricati {len(audio_list)} file audio dalla cartella {cartella}.")
    return audio_list
carica_audio_mono(original_dir)

# Cell 15
'''def carica_audio_mono_e_visualizza(cartella):
    filepaths = glob(os.path.join(cartella, '*.wav'))
    audio_list = []
    
    for fp in filepaths:
        loader = es.MonoLoader(filename=fp)
        audio = loader()
        audio_list.append((fp, audio))
        print(f"Caricato file: {fp}, durata: {len(audio)/sr:.2f} secondi")
        
        # Visualizza il file audio con IPython.display
        display(Audio(data=audio, rate=sr))
    
    print(f"Caricati {len(audio_list)} file audio dalla cartella {cartella}.")
    return audio_list
carica_audio_mono_e_visualizza(original_dir)'''

# Cell 16
# Funzione per suddividere un segnale audio in grani
def suddividi_in_grani(audio, grain_size=1024, overlap=512):
    step = grain_size - overlap
    grains = [audio[i:i+grain_size] for i in range(0, len(audio) - grain_size, step)]
    return np.array(grains)

# Cell 17
# Get the first audio file from the original directory
audio_files = carica_audio_mono(original_dir)
if audio_files:
    filename, audio = audio_files[0]
    
    # Generate grains using the existing function and global parameters
    grains = suddividi_in_grani(audio, grain_size=window, overlap=hop)
    
    print(f"Number of grains generated: {len(grains)}")
    print(f"Each grain size: {window} samples")
    
    # Display first 5 grains as audio
    print("\nPlaying first 5 grains:")
    for i, grain in enumerate(grains[:5]):
        print(f"\nGrain {i+1}:")
        display(Audio(data=grain, rate=sr))

# Cell 18
def ricombina_grani(grani, grain_size, overlap, shuffle=True):
    if shuffle:
        np.random.shuffle(grani)
    output_length = (len(grani) - 1) * (grain_size - overlap) + grain_size
    audio_out = np.zeros(output_length)
    for i, grain in enumerate(grani):
        start = i * (grain_size - overlap)
        audio_out[start:start+grain_size] += grain
    return audio_out

# Cell 19
""" from IPython.display import Audio, display

def visualizza_audio_elaborati(audio_list, sr):

    for i, (filename, audio) in enumerate(audio_list):
        print(f"File elaborato {i+1}: {filename}, durata: {len(audio)/sr:.2f} secondi")
        display(Audio(data=audio, rate=sr))

# Esempio di utilizzo con file elaborati
# Supponiamo di avere una lista di file elaborati
audio_elaborati = []

# Carica un file audio, suddividilo in grani e ricombinalo
audio_files = carica_audio_mono(original_dir)
if audio_files:
    filename, audio = audio_files[0]
    
    # Suddividi in grani
    grains = suddividi_in_grani(audio, grain_size=window, overlap=hop)
    
    # Ricombina i grani
    audio_ricombinato = ricombina_grani(grains, grain_size=window, overlap=hop, shuffle=True)
    
    # Aggiungi il file ricombinato alla lista
    audio_elaborati.append((f"Ricombinato_{filename}", audio_ricombinato))

# Visualizza i file elaborati
visualizza_audio_elaborati(audio_elaborati, sr) """

# Cell 20
def salva_audio(audio, path):
    writer = es.MonoWriter(filename=path)
    writer(audio)


# Cell 21
# Funzione principale per elaborare tutti i file

def elabora_cartella_audio(input_dir='dataset/originali', output_dir='dataset/granulati', grain_size=default_grain_size, overlap=default_overlap):
    os.makedirs(output_dir, exist_ok=True)
    audio_pairs = []
    audio_list = carica_audio_mono(input_dir)

    for path, audio in audio_list:
        grani = suddividi_in_grani(audio, grain_size, overlap)
        audio_granulato = ricombina_grani(grani, grain_size, overlap)
        nome_file = os.path.basename(path)
        output_path = os.path.join(output_dir, 'granulato_' + nome_file)
        salva_audio(audio_granulato, output_path)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=audio, mode='lines', name='Originale'))
        fig.add_trace(go.Scatter(y=audio_granulato,opacity=0.5, mode='lines', name='Granulato'))
        fig.update_layout(title=f'Onda sonora: {nome_file}', xaxis_title='Campione', yaxis_title='Ampiezza')
        

        audio_pairs.append((audio, audio_granulato))

    return audio_pairs

# Cell 22
elabora_cartella_audio('dataset/originali', 'dataset/granulati', default_grain_size, default_overlap)

# Cell 23
def carica_dataset_audio_granulato(originali_dir='dataset/originali', granulati_dir='dataset/granulati'):
    X, y = [] , []
    for nome_file in os.listdir(originali_dir):
        if nome_file.endswith('.wav'):
            path_originale = os.path.join(originali_dir, nome_file)
            path_granulato = os.path.join(granulati_dir, f'granulato_{nome_file}')

            if os.path.exists(path_granulato):
                loader = es.MonoLoader(filename=path_originale)
                audio_originale = loader()

                loader = es.MonoLoader(filename=path_granulato)
                audio_granulato = loader()

                X.append(audio_granulato)
                y.append(audio_originale)

    max_len = max(max(len(x), len(t)) for x, t in zip(X, y))

    X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    y_padded = np.array([np.pad(t, (0, max_len - len(t)), 'constant') for t in y])

    return X_padded, y_padded

# Funzione per salvare X e y come file .npz

def salva_dataset_npz(X, y, percorso='dataset/dataset.npz'):
    np.savez(percorso, X=X, y=y)
    print(f"Dataset salvato in {percorso}")

# Cell 24
def test_carica_dataset():
    # Load the dataset
    X, y = carica_dataset_audio_granulato()
    salva_dataset_npz(X, y, 'dataset/dataset.npz')
    
    print(f"Dataset loaded: {len(X)} pairs of audio files")
    
    # Create visualization for the first few samples
    for i in range(min(3, len(X))):
        # Create subplot with two traces
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Audio Granulato (Input)', 
                                         'Audio Originale (Target)'))
        
        # Add granulated audio trace
        fig.add_trace(
            go.Scatter(y=X[i], mode='lines', name='Granulato'),
            row=1, col=1
        )
        
        # Add original audio trace
        fig.add_trace(
            go.Scatter(y=y[i], mode='lines', name='Originale'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"Coppia Audio #{i+1}",
            showlegend=True
        )
        fig.show()
        
        # Display audio players
        print(f"\nCoppia Audio #{i+1}")
        print("Audio Granulato:")
        display(Audio(data=X[i], rate=sr))
        print("Audio Originale:")
        display(Audio(data=y[i], rate=sr))
test_carica_dataset()

# Cell 25
def crea_modello(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(2048, activation='relu'),
        Dense(2048, activation='relu'),
        Dense(input_dim)  # uscita = stessa forma input (regressione)
    ])
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy'
        )
    return model


# Cell 26
def aggiungi_rumore(audio, livello_rumore=0.005):
    rumore = np.random.normal(0, livello_rumore, audio.shape)
    return audio + rumore


# Cell 27
default_epoche=1000
default_batch_size=32

# Cell 28
def addestra_modello(X, y, epoche=50, batch_size=32):
    input_dim = X.shape[1]
    model = crea_modello(input_dim)

    # Augmentazione dati (opzionale)
    #X_aug = np.array([aggiungi_rumore(x) for x in X])
    #history = model.fit(X_aug, y, epochs=epoche, batch_size=batch_size, validation_split=0.1)

    history = model.fit(X, y, epochs=epoche, batch_size=batch_size, validation_split=0.1)

    print("Modello addestrato. Pesi del modello:")
    for layer in model.layers:
        pesi = layer.get_weights()
        for i, peso in enumerate(pesi):
            print(f"Layer {layer.name} - Peso {i}:", peso.shape)
            print(peso)

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Andamento Loss durante il training')
    plt.show()

    return model


# Cell 29
def mostra_audio_risultato(model, X, y, indice=0):
    #pred = model.predict(np.array([X[indice]]))[0]
    prob = Sequential([
    model,
    Softmax()
    #applico softmax per avere le probabilità
    ])
    predict = prob.predict(np.array([X[indice]]))[0]
    print("Predizione:", predict[0])
    plt.figure(figsize=(12, 4))
    plt.plot(y[indice], label='Originale')
    plt.plot(pred, label='Predetto')
    plt.legend()
    plt.title("Audio Originale vs Predetto")
    plt.show()

    print("Audio Predetto:")
    display(Audio(pred, rate=sr))

    print("Audio Originale:")
    display(Audio(y[indice], rate=sr))

# Cell 30
# Load the dataset
data = np.load('dataset/dataset.npz')
X, y = data['X'], data['y']

print("Dataset shape:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Create and train the model
input_dim = X.shape[1]
model = crea_modello(input_dim)

# Data augmentation with noise
#X_aug = np.array([aggiungi_rumore(x) for x in X])

# Training with validation split
history = model.fit(
    X, 
    y, 
    epochs=default_epoche, 
    batch_size=default_batch_size, 
    validation_split=0.2,
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Debug results on first 3 samples
print("\nVisualizing results for first 3 samples:")
for i in range(min(3, len(X))):
    print(f"\nSample {i+1}:")
    mostra_audio_risultato(model, X, y, i)


# Cell 31
model.summary()

# Cell 32
""" # Load dataset
data = np.load('dataset/dataset.npz')
X, y = data['X'], data['y']

print("Dataset caricato:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Train model
model = addestra_modello(X, y, epoche=default_epoche, batch_size=default_batch_size, salva_pesi=default_model_folder)

# Visualize results for first 3 samples
for i in range(min(3, len(X))):
    print(f"\nRisultati per il campione {i+1}:")
    mostra_audio_risultato(model, X, y, i) """

# Cell 33
def predici_audio(model, audio_input):
    audio_input = np.array(audio_input)
    if len(audio_input.shape) == 1:
        audio_input = np.expand_dims(audio_input, axis=0)
    return model.predict(audio_input)[0]

def visualizza_audio(audio_granulato, audio_rigenerato, audio_originale=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=audio_granulato, opacity=0.5, mode='lines', name='Granulato'))
    fig.add_trace(go.Scatter(y=audio_rigenerato, mode='lines', name='Rigenerato'))
    if audio_originale is not None:
        fig.add_trace(go.Scatter(y=audio_originale, opacity=0.5, mode='lines', name='Originale'))
    fig.update_layout(title='Confronto Audio', xaxis_title='Campioni', yaxis_title='Ampiezza')
    fig.show()

def interfaccia_audio(model, X, y=None):
    def aggiorna(indice):
        audio_granulato = X[indice]
        audio_rigenerato = predici_audio(model, audio_granulato)
        audio_originale = y[indice] if y is not None else None

        visualizza_audio(audio_granulato, audio_rigenerato, audio_originale)

        print("Audio granulato:")
        display(Audio(audio_granulato, rate=sr))

        print("Audio rigenerato:")
        display(Audio(audio_rigenerato, rate=sr))

        if audio_originale is not None:
            print("Audio originale:")
            display(Audio(audio_originale, rate=sr))

    slider = widgets.IntSlider(min=0, max=len(X)-1, step=1, description='Esempio')
    widgets.interact(aggiorna, indice=slider)

# Cell 34
interfaccia_audio(model, X, y)

