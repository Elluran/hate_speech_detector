import sys
import time
import warnings
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = RobertaForSequenceClassification.from_pretrained("Elluran/Hate_speech_detector")
tokenizer = RobertaTokenizerFast.from_pretrained("Elluran/Hate_speech_detector")
THRESHOLD = 0.810126582278481

def clean_string(s):
    s = s.encode('ASCII', 'ignore').decode()
    s = s.lower()
    s = ''.join([ch for ch in s if ch.isalpha() or ch == ' '])
    s = " ".join(s.split())
    if len(s) > 0 and s[-1] == ' ':
        s = s[:-1]
    if len(s) > 0 and s[0] == ' ':
        s = s[1:] 
    return s

if __name__ == "__main__":
    start_time = time.time()
    input_string = clean_string(sys.argv[1])
    tokens = tokenizer(input_string, padding=True, truncation=True,  return_tensors="pt")
    output = model(**tokens)
    result = "hate speech" if torch.sigmoid(output[0]) > THRESHOLD else "regular tweet"
    if torch.sigmoid(output[0]) > THRESHOLD:
        confidence = torch.sigmoid(output[0]).item()
    else:
        confidence =  1 - torch.sigmoid(output[0]).item()
    
    print(f"answer is: {result} with confidence: {'{:.2f}'.format(confidence)}")
    print(time.time() - start_time, "time spent for tweet")