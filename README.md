import pandas as pd  
import matplotlib.pyplot as plt  
import spacy  
from spacy.util import minibatch, compounding  
import random  
 
nlp = spacy.load('el__core__news__md')  
df1 = pd.read__csv('../data/jtp__fake__news.csv')  
df1.replace(to__replace='[ \ n \ r \ t]', value=' ', regex=True, inplace=True)  
 
def load__data(train__data, limit=0, split=0.8):  
    random.shuffle(train__data)  
    train__data = train__data[-limit:]  
    texts, labels = zip(*train__data)  
    cats = [{"REAL": not bool(y), "FAKE": bool(y)} for y in labels]  
    split = int(len(train__data) * split)  
     
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])  
# - - - - - - - - - - - - - - - - - - evaluate function defined below- - - - - - - - - - - -  
def evaluate(tokenizer, textcat, texts, cats):  
    docs = (tokenizer(text) for text in texts)  
    tp = 0.0  # True positives  
    fp = 1e-8  # False positives  
    fn = 1e-8  # False negatives  
    tn = 0.0  # True negatives  
    for i, doc in enumerate(textcat.pipe(docs)):  
        gold = cats[i]  
        for the label, score in doc.cats.items():  
            if the label is not in gold:  
                continue  
            if label = = "FAKE":  
                continue  
            if score > = 0.5 and gold[label] > = 0.5:  
                tp + = 1.0  
            elif score > = 0.5 and gold[label] < 0.5:  
                fp + = 1.0  
            elif score < 0.5 and gold[label] < 0.5:  
                tn + = 1  
            elif score < 0.5 and gold[label] > = 0.5:  
                fn + = 1  
    precision = tp / (tp + fp)  
    recall = tp / (tp + fn)  
#- - - - - - - - - - - -if conditions for precision recall - - - - - - - - -  
    if (precision + recall) = = 0:  
        f__score = 0.0  
    else:  
        f__score = 2 * (precision * recall) / (precision + recall)  
    return {"textcat__p": precision, "textcat__r": recall, "textcat__f": f__score}  
In [3]:  
df1.info()  
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 100 entries, 0 to 99  
Data columns (total five columns):  
 #   Column   Non-Null Count  Dtype  
--  -   - - - - - -      - - - - - - - - - - - - -  - - - - -  
 0   title    100 non-null    object  
 One text     100 non-null    object  
 Two sources 100 non-null    object  
 Three url      100 non-null    object  
 4   is__fake  100 non-null    int64  
dtypes: int64(1), object(4)  
memory usage: 4.0+ KB  
textcat=nlp.create__pipe( "textcat", config={"exclusive__classes": True, "architecture": "simple__cnn"})  
nlp.add__pipe(textcat, last=True)  
nlp.pipe__names  
['tagger', 'parser', 'ner', 'textcat']  
textcat.add__label("REAL")  
textcat.add__label("FAKE")  
df1['tuples'] = df1.apply(lambda row: (row['text'], row['is__fake']), axis=1)  
train = df1['tuples'].tolist()  
(train__texts, train__cats), (dev__texts, dev__cats) = load__data(train, split=0.9)  
 
train__data = list(zip(train__texts,[{'cats': cats} for cats in train__cats]))  
n__iter = 20  
# - - - - - - - - - - - - Disabling other components- - - - - - - - - - - - -    
other__pipes = [pipe for pipe in nlp.pipe__names if pipe != 'textcat']  
with nlp.disable__pipes(*other__pipes):  # only train textcat  
    optimizer = nlp.begin__training()  
 
    print("Training the model...")  
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))  # fakenews
