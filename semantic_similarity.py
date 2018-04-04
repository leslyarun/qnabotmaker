import spacy
import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_lg')

df = pd.read_table(r"C:\Users\krgp037\Downloads\qnamaker.tsv")
jdf = pd.read_json(r"C:/Users/krgp037/Downloads/sample_qnajson.json")

def calculateScore(df, msg):
    sim = list()
    chat = nlp(msg)

    for v in df['Question']:
        qs = nlp(v)
        sim.append(qs.similarity(chat))

    similarity = np.array(sim)
    max_prob = similarity.argmax()
    prob = similarity[max_prob]

    if prob >= 0.70:
        print('User Qs. is -->', chat.text)
        print('Matching Qs. is -->', df['Question'][max_prob])
        print('Matching Qs. Probability is -->', prob)
        print('Bot Answer is -->', df['Answer'][max_prob])

    else:
        print("Sorry, I'm still learning :) Please reframe your question :)")

def calculateJsonScore(df, msg):
    sim = list()
    chat = nlp(msg)

    pairs = df['pairs']

    for pair in pairs:
        ques = (pair['questions'])
        question = " ".join(ques)
        qspair = nlp(question)
        sim.append(qspair.similarity(chat))

    similarity = np.array(sim)
    max_prob = similarity.argmax()
    prob = similarity[max_prob]

    if prob >= 0.70:
        print('User Qs. is -->', chat.text)
        print('Matching Qs. is -->', df['pairs'][max_prob]['questions'])
        print('Matching Qs. Probability is -->', prob)
        print('Bot Answer is -->', df['pairs'][max_prob]['answer'])

    else:
        print("Sorry, I'm still learning :) Please reframe your question :)")

calculateScore(df, 'hi')
calculateJsonScore(jdf, 'i want to change joining date')