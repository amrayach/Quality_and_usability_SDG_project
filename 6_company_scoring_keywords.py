import pickle
import pandas as pd

# Loading the keyword results from a the previously generated pickle file
with open('keyword_result_per_sus_report.pickle', 'rb') as f:
    loaded_obj = pickle.load(f)

# Creating a dictionary to store the results
data = {'Name': [],
        'Score': [],
        'Normalized Score': []
        }

# for each report calculated score in the pickle file
for reportName in loaded_obj:
    report = loaded_obj.get(reportName)
    data['Name'].append(reportName)  # We extract the name
    sum = 0
    weights = report.get('keyword_occurences_times_weights')  # Get the weights of each SDG keywods
    for sdg in weights:
        sum += weights.get(sdg)  # add all the weights
    Score = sum / report.get('num_of_tokens')  # divide the result by the number of words
    data['Score'].append(Score)  # add the score


Score_min = min(data['Score'])  # calculate score min
Score_max = max(data['Score'])  # calculate score max

# We normalize the scores and add a factor 10
# so now we get scores from 0-10

for score in data['Score']:
    data['Normalized Score'].append(((score - Score_min) / (Score_max - Score_min)) * 10)

df = pd.DataFrame(data)
df.to_csv('company_score_keywords.csv', index=False)
