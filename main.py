import numpy as np
import pymorphy2
import string
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from PIL import Image


def main():
    file_name = "test.txt"
    file = open(file_name, "r", encoding="utf-8")
    text = file.read()
    text = process_text(text)
    text = normalise_text(text)
    generate_cloud(text)


def process_text(text):
    nltk.download('stopwords')
    text = text.lower()
    special_chars = string.punctuation + '\n\xa0«»\t—–…' + string.digits
    r_stopwords = stopwords.words("russian")
    text = "".join([character if character not in special_chars else " " for character in text])
    text = text.split()
    text = filter(lambda x: x not in r_stopwords, text)
    return text


def normalise_text(text):
    morph = pymorphy2.MorphAnalyzer()
    text = " ".join(list(map(lambda x: morph.parse(x)[0].normal_form, text)))
    return text


def generate_cloud(text):
    mask = np.array(Image.open("mask.jpg"))
    cloud = WordCloud(mask=mask).generate(text)
    cloud.to_file("picture.png")


if __name__ == '__main__':
    main()
