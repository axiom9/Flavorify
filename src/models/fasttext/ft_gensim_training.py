from gensim.models import FastText
import importlib


import fasttext_prep
importlib.reload(fasttext_prep)

model_dir = 'ft_model/'
df = fasttext_prep.load_gensim()
df_train, df_test = fasttext_prep.prep_gensim(df)

embedding_size=60
window_size=30
min_word=3
down_sampling=1e-2
epochs=5

data=df_train.Text #training only on the training data

try:
    print(f'Attempting to load model from {model_dir+"ft_model.model"}')
    ft_model = FastText.load(model_dir+'ft_model.model')
except:
    print('Model loading failed-- Training new model')
    ft_model = FastText(sentences=data,
                        vector_size=embedding_size,
                        window=window_size,
                        min_count=min_word,
                        sample=down_sampling,
                        epochs=epochs,
                        sg=1)
    print(f'Model training completed, saving model to: {model_dir +"ft_model.model"}')
    ft_model.save(model_dir+'ft_model.model')

