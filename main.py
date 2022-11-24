import numpy as np
import sys
import pymorphy2
import string
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from PIL import Image


def main():
    try:
        script_file_name, mask_name = sys.argv[1:3]
    except ValueError:
        raise SystemExit(f"Usage: {sys.argv[0]} <script_file_name> <mask_name>")
    file = open(script_file_name, "r", encoding="utf-8")
    text = file.read()
    file.close()
    text = process_text(text)
    text = normalise_text(text)
    generate_cloud(text, mask_name)


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

    def normaliser(word):
        return morph.parse(word)[0].normal_form
    text = " ".join(list(map(normaliser, text)))
    return text


def generate_cloud(text, mask_name):
    mask = np.array(Image.open(mask_name))
    cloud = WordCloud(mask=mask).generate(text)
    cloud.to_file("picture.png")


if __name__ == '__main__':
    main()
