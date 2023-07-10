import fitz
import os
import pandas as pd
import spacy
import re

# load spacy's  large english nlp model "en_core_web_lg"
# disable any pipeline component that is not needed for
# sentence segmentation, to make the process faster
# The model should be already be downloaded through the requirements.txt file
# if not, run the following command in the terminal:
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "morphologizer",
                                            "tagger", "attribute_ruler"])

# define input and output paths

# This is the path to the folder where the pdf files are stored
# that should be processed and converted to text files
input_pdf_path = 'Sustainability_Reports_PDF'

# This is the path to the folder where the processed/extracted text files should be stored
output_text_path = 'Sustainability_Reports_TXT'

# check if the output folder exists, if not create it
if not os.path.exists(output_text_path):
    os.makedirs(output_text_path, exist_ok=True)

# iterate through the pdf under the folder Sustainability_Reports_PDF
# if file is pdf:
# 1) extract the text per page
# 2) run some processing (splitting after "\n", striping, removing sentences shorter than 2 tokens)
# 3) convert to dataframe: remove empty rows and remove duplicate sentences
# 4) convert list of sentences to a long text and run spacy's nlp over it
# 5) extract the detected sentences from spacy's senter model (https://spacy.io/api/sentencerecognizer)
# 6) apply last processing steps like filtering short sentences (less than two words)
# 7) save it to txt file for later use

# iterate through the pdf files in the input folder
for filename in os.listdir(input_pdf_path):
    # check if the file is a pdf file
    if filename.endswith('.pdf'):
        print("Starting to process file: " + filename + " ...")

        # open the pdf file with fitz (PyMuPDF) and extract the text per page
        doc = fitz.open(input_pdf_path + '/' + filename)

        # create an empty list to store the text per page
        text = []

        # iterate through the pages of the pdf file
        for page in doc:
            # extract the text per page and split it after "\n"
            text.extend(page.get_text().split("\n"))

        # remove empty strings from the list, strip the strings and remove sentences shorter than 2 tokens
        # this is necessary because the pdf extraction sometimes returns empty strings
        # join the list to a long string
        text = " ".join(list(map(lambda x: " ".join(x), list(
            filter(lambda x: len(x) > 3, list(map(lambda x: re.sub(' +', ' ', x).strip().split(" "), text)))))))

        # run spacy's nlp over the long string
        curr_doc = nlp(text)
        # extract the detected sentences from spacy's senter model (https://spacy.io/api/sentencerecognizer)
        text = [sent.text for sent in curr_doc.sents]

        # apply last processing steps like filtering short sentences (less than two words)
        text = list(map(lambda x: " ".join(x), list(
            filter(lambda x: len(x) > 3, list(map(lambda x: re.sub(' +', ' ', x).strip().split(" "), text))))))

        # convert list to pandas dataframe
        df = pd.DataFrame(text)
        # remove rows with empty string
        df = df[df[0] != '']
        # remove duplicate rows
        df = df.drop_duplicates()
        # convert dataframe to list and join with "\n"
        text = "\n".join(df[0].tolist())

        # save extracted sentences as text file in
        with open(output_text_path + '/' + filename[:-4] + '.txt', 'w') as f:
            print("writing txt file: " + filename)
            f.write(text)
