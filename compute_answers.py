import sys
import json
from tensorflow.keras.models import load_model
from preprocessor import preprocessor
import numpy as np


MODEL_FP = "models/squad-model.h5"
WORD_TOKENIZER_FP = "SQUAD MATERIAL/word_tokenizer.pkl"
NER_TOKENIZER_FP = "SQUAD MATERIAL/ner_tokenizer.pkl"
POS_TOKENIZER_FP = "SQUAD MATERIAL/pos_tokenizer.pkl"

input_filepath = sys.argv[1]
model = load_model(MODEL_FP)
paragraph_maxlen = model.layers[0].input_shape[0][1]
question_maxlen = model.layers[1].input_shape[0][1]
preprocessor = preprocessor(input_filepath,WORD_TOKENIZER_FP,NER_TOKENIZER_FP,POS_TOKENIZER_FP,paragraph_maxlen,question_maxlen)
X_input = preprocessor.get_model_input()
#X_input = load(open('X_input.pkl', 'rb'))
predictions = model.predict(X_input)
answers = preprocessor.map_predictions(np.argmax(predictions[0],axis=-1),np.argmax(predictions[1],axis=-1)) 
with open("predictions.json", "w") as outfile:  
    json.dump(answers, outfile)