import pandas as pd
import pickle
import os
from flair.data import Sentence
from flair.models import TextClassifier
from huggingface_hub import hf_hub_download

# 1) This script calculates keyword occurrences in the sustainability reports on a document level using the SDG keywords from
# the university of auckland
# 2) The script also runs the flair model on the sustainability reports and classifies the documents on a sentence level for SDG's


####################################
# 1) Calculate keyword occurrences #
####################################

# load keywords from file "UoA-SDG-Keyword-List-Ver.-1.1.xlsx" retrieved from the university of auckland
# The keywords are downloaded from the following link: https://www.sdgmapping.auckland.ac.nz/files/2020/10/UoA-SDG-Keyword-List-Ver.-1.1.xlsx
# load the data from the excel file, using the pandas excel reader

df = pd.read_excel("UoA-SDG-Keyword-List-Ver.-1.1.xlsx")

# create a list of keywords for SDG 17, since the excel file does not contain the keywords for SDG 17
sdg_17 = ['Capacity building', 'Civil society partnerships', 'Communication technologies', 'Debt sustainability',
          'Development assistance', 'Disaggregated data', 'Doha Development Agenda', 'Entrepreneurship',
          'Environmentally sound technologies', 'Foreign direct investments', 'Fostering innovation', 'Free trade',
          'Fundamental principles of official statistics', 'Global partnership',
          'Global partnership for sustainable development', 'Global stability', 'International aid',
          'International cooperation', 'International population and housing census', 'International support',
          'International support for developing countries', 'Knowledge sharing', 'Multi-stakeholder partnerships',
          'Poverty eradication', 'Public-private partnerships', 'Science cooperation agreements',
          'Technology cooperation agreements', 'Technology transfer', 'Weighted tariff average', 'Women entrepreneurs',
          'World Trade Organization']

# initialize the keywords map with the keywords for SDG 17
keywords_map = {"SDG17": sdg_17}
# iterate through the SDG labels and initialize the keywords map with empty lists
# the lists will be filled with the keywords for each SDG from the excel file
for sdg_label in list(set(df["SDG"].to_list())):
    keywords_map[sdg_label] = []

# iterate through the rows of the excel file and add the keywords to the keywords map
# use the following columns: "SDG Keywords", "Alternatives", "SDG"
# using the default zip function, we can iterate through the three columns simultaneously
for (sdg_keyword, sdg_alternative, sdg_label) in zip(df["SDG Keywords"].to_list(), df["Alternatives"].to_list(),
                                                     df["SDG"].to_list()):
    # check if the keyword is a string or nan
    if type(sdg_alternative) == str:
        # some keywords have multiple alternatives, separated by ";" in the excel file
        # split the alternatives and add them to the keywords map using extend()
        keywords_map[sdg_label].extend(sdg_alternative.strip().split(";"))

    # add the keyword to the keywords map
    keywords_map[sdg_label].extend([sdg_keyword])

# since the keywords of each SDG are not equally distributed, we need to weight the keywords
# the weight is calculated by dividing the number of keywords for each SDG by the total number of keywords
# the weight is used to calculate the number of occurrence of each SDG-keyword in a text file
keyword_freq_weights = {}
for sdg_label in keywords_map.keys():
    keyword_freq_weights[sdg_label] = len(keywords_map[sdg_label]) / sum(
        [len(keywords_map[i]) for i in list(keywords_map.keys())])

# iterate through text files in folder "Sustainability_Reports_TXT"
txt_sdg_count = {}
for sus_text_file in os.listdir("./Sustainability_Reports_TXT"):
    # check if the file is a text file
    if sus_text_file.endswith(".txt"):
        print("Processing file: ", sus_text_file)
        # read the text file
        with open("./Sustainability_Reports_TXT/" + sus_text_file, "r") as file:
            text = file.read()

        # create a dict that counts the number of keywords for each SDG
        # in each text file
        sdg_count = {}

        # initialize the dict with 0 for each SDG
        for sdg_label in keywords_map.keys():
            sdg_count[sdg_label] = 0

        total_count = 0

        # iterate through the keywords and count the number of occurrences
        for sdg_label in keywords_map.keys():
            for keyword in keywords_map[sdg_label]:
                # count the number of occurrences of the keyword in the text file
                sdg_count[sdg_label] += text.count(keyword)
                # increase the total count of keywords in the text file
                total_count += text.count(keyword)

        # multiply the number of occurrences of each keyword with the weight of the keyword
        for sdg_count_curr in sdg_count.keys():
            sdg_count[sdg_count_curr] = float(sdg_count[sdg_count_curr]) * float(keyword_freq_weights[sdg_label])

        # add the result to the main dict
        txt_sdg_count[sus_text_file] = {"keyword_occurences_times_weights": sdg_count,
                                        "num_of_tokens": len(text.split(" "))}

# save the result to a pickle file
# the pickle file is used in a later script to compute the company SDG score 
with open('keyword_result_per_sus_report.pickle', 'wb') as f:
    pickle.dump(txt_sdg_count, f)

#########################################
# 2) Calculate sentence classifications #
#########################################

# check if the model is already downloaded
# if not, download the model from huggingface hub
if not os.path.exists("Model/final-model.pt"):
    hf_hub_download(repo_id="amay01/quality_and_usability_sgd_17_classifier_flair", filename="final-model.pt",
                    local_dir="./Model/")

# load the model
model = TextClassifier.load('Model/final-model.pt')

# This directory contains the extracted text files of the sustainability reports
sus_dir = "./Sustainability_Reports_TXT/"

# This directory will contain the sentence classifications for each sustainability report
sus_model_sent_prediction_dir = "./Sustainability_model_predictions/"

# check if the directory exists
# if not, create the directory
if not os.path.exists(sus_model_sent_prediction_dir):
    os.makedirs(sus_model_sent_prediction_dir, exist_ok=True)

# iterate through the text files in the directory
for sus_text_file in os.listdir(sus_dir):
    # check if the file is a text file
    if sus_text_file.endswith(".txt"):
        print("Processing file: ", sus_text_file)

        # read the text file
        # split the text file into sentences using "\n" as delimiter
        with open(sus_dir + sus_text_file, "r") as file:
            text = file.read().split("\n")

        # initialize the list that will contain the sentence classifications
        sent_prediction = []

        # iterate through the sentences and classify them
        for sent in text:
            print("Processing sentence: ", sent)
            # create a compatible flair sentence object
            curr_sent = Sentence(sent)
            # predict the label of the sentence
            model.predict(curr_sent)

            # if the model predicts a label with a score higher than 0.55, add the label to the list
            # if the score is lower than 0.55, add the label "NONE" to the list
            # if the model does not predict a label, add the label "NONE" to the list
            if curr_sent.labels:
                if curr_sent.score <= 0.55:
                    sent_prediction.append([curr_sent.text, "__label__NONE"])
                else:
                    sent_prediction.append([curr_sent.text, curr_sent.tag])
            else:
                sent_prediction.append([curr_sent.text, "__label__NONE"])

        # create a pandas dataframe from the list, with the columns "sentence" and predicted "sdg_label"
        df = pd.DataFrame(sent_prediction, columns=['sentence', 'sdg_label'])
        # save the dataframe to a csv file in the directory "Sustainability_model_predictions"
        df.to_csv(sus_model_sent_prediction_dir + sus_text_file.replace(".txt", ".csv"), index=False)
