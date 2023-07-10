import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from huggingface_hub import hf_hub_download
import csv


# We have four sdg labeled datasets with 17 classes each:
# 1. Original Dataset (original_data.csv)
# 2. politics.csv
# 3. targets.csv
# 4. Community gathered dataset (osdg_v2023.csv)
#    taken from (https://osdg.ai/community)
############################################

# For the classification task, we'll use 3 approaches to merge the datasets:
# 1. Merge all the datasets together
# 2. Merge the original dataset with the politics and targets datasets
# 3. Use the original dataset only
############################################

# For the classification task, following transformer models could be tested:
# 1. bert-base-uncased
# 2. bert-base-cased
# 3. roberta-base
# 4. jonas/bert-base-uncased-finetuned-sdg-Mar23
# 5. jonas/bert-base-uncased-finetuned-sdg
# 6. sadickam/sdg-classification-bert
# Note: 4, 5, 6 are fine-tuned on the osdg dataset already, so they'll be
#       fine-tuned on any dataset variation that does not include the osdg dataset
############################################

# The datasets need to be preprocessed to be compatible with the flair library
# The preprocessing steps are as follows:
# 1. Convert the SDG column to integers
# 2. Subtract 1 from the SDG column to make it 0-indexed
# 3. Rename the columns to label and text
# 4. Optional: Convert the texts to lowercase
# 5. Add the label prefix to the label column to make it compatible with flair
############################################

# Finally the datasets need to be split into train, dev, and test sets
# The split is done using StratifiedKFold with 5 splits
# The splits are saved in the Data folder
############################################


def convert_dataset_flair_format(csv_path):
    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError("The file does not exist")

    # since the datasets are in different formats, we need to check which dataset is being used
    # to bring them to the same format by dropping the unnecessary columns
    # and renaming the columns to label and text
    # and making the labels 0-indexed
    # before converting them to the flair format
    if csv_path == "Data/original_data.csv":
        df = pd.read_csv(csv_path)
        # drop 2,3,4 columns
        df = df.drop(df.columns[[2, 3, 4]], axis=1)
        # convert the SDG colum to integers
        df['SDG'] = df['SDG'].astype(int)
        # subtract 1 from the SDG column to make it 0-indexed
        df['SDG'] = df['SDG'] - 1
        # rename the columns to label and text
        df = df[['SDG', 'Text']].rename(columns={"SDG": "label", "Text": "text"})

    elif csv_path == "Data/politics.csv":
        # read in the data from the csv file, with "," delimiter:
        df = pd.read_csv(csv_path, delimiter=',')
        df = df.reindex(columns=["label", "text"])
        df['label'] = df['label'].astype(int)
        # subtract 1 from the SDG column to make it 0-indexed
        df['label'] = df['label'] - 1

    elif csv_path == "Data/targets.csv":
        # read in the data from the csv file, with "," delimiter:
        df = pd.read_csv(csv_path, sep=',')
        # drop nan values
        df = df.dropna()

        df = df[['text', 'sdg']].rename(columns={"text": "text", "sdg": "label"})
        df = df.reindex(columns=["label", "text"])
        df['label'] = df['label'].astype(int)
        # subtract 1 from the SDG column to make it 0-indexed
        df['label'] = df['label'] - 1

    elif csv_path == "Data/osdg_v2023.csv":
        # since pandas is failing to handle double quotes in the csv file
        # we'll read in the data from the csv file and iterate over lines:
        csv_file = open(csv_path, 'r').read().split('\n')[1:-1]

        # extract the text and label from each line
        label_text = []
        for i in csv_file:
            i = i.split("\t")
            label_text.append([int(i[3]) - 1, i[2]])

        # create a dataframe from the text and label
        df = pd.DataFrame(label_text, columns=['label', 'text'])

    else:
        raise FileNotFoundError("The Dataset File/Format is not supported")

    return df


def create_kfold_stratified_splits(df, output_path):
    # convert labels to flair format __label__ + sdg number
    df['label'] = '__label__' + df['label'].astype(str)

    # initialize the stratified k-fold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # split the data into train and test sets
    # using the stratified k-fold object with the text, label columns enumerated
    for counter, idx in enumerate(skf.split(df['text'], df['label'])):
        # extract the train and test sets using the indices
        train_df, test_df = df.iloc[idx[0]], df.iloc[idx[1]]

        # further split the train set into train and dev sets
        train_df, dev_df = train_test_split(train_df, test_size=0.05)

        # print the label distribution for each set
        label_freq_train = train_df['label'].value_counts()
        label_freq_dev = dev_df['label'].value_counts()
        label_freq_test = test_df['label'].value_counts()

        print("Train Set Label Distribution: \n", label_freq_train)
        print("Dev Set Label Distribution: \n", label_freq_dev)
        print("Test Set Label Distribution: \n", label_freq_test)
        print("Total: ", label_freq_train.sum() + label_freq_dev.sum() + label_freq_test.sum())

        # create the directory for the data
        os.makedirs("./" + output_path + "/" + str(counter + 1) + "/", exist_ok=True)

        # save the data to the directory as csv files
        train_df.to_csv("./" + output_path + "/" + str(counter + 1) + "/train.csv", sep='\t', index=False, header=False)
        dev_df.to_csv("./" + output_path + "/" + str(counter + 1) + "/dev.csv", sep='\t', index=False, header=False)
        test_df.to_csv("./" + output_path + "/" + str(counter + 1) + "/test.csv", sep='\t', index=False, header=False)


def main(datasets):
    # create an empty list to store the later on converted dataframes
    dfs = []

    # create an empty 2x2 subplot for the label distribution for each dataset
    fig, axs = plt.subplots(2, 2)

    # Iterate over the datasets in the Data folder
    # visualize the label distribution as a bar plot in a (2,2) subplot
    # sorted by the label index
    # y-axis is the number of samples
    # x-axis is the label index
    for file in datasets:
        # convert the dataset to flair format
        # more information about the format can be found under the function
        df = convert_dataset_flair_format("Data/" + file)

        # visualize the label distribution as a bar plot
        # sorted by the label index
        # y-axis is the number of samples

        # sort the labels by index, set the x,y labels and the title for the respective subplots
        df['label'].value_counts().sort_index().plot(kind='bar',
                                                     ax=axs[datasets.index(file) // 2, datasets.index(file) % 2])
        axs[datasets.index(file) // 2, datasets.index(file) % 2].set_xlabel("Label Index")
        axs[datasets.index(file) // 2, datasets.index(file) % 2].set_ylabel("Number of Samples")
        axs[datasets.index(file) // 2, datasets.index(file) % 2].set_title(file.replace(".csv", ""))

        # append the converted dataframe to the list
        dfs.append(df)

    # adjust the spacing between the subplots
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)

    # remove the x ticks from the subplots
    # remove y title from the subplots
    # only keep the x title for the bottom subplots
    # only keep the y title for the left subplots

    for ax in axs.flat:
        ax.set(xlabel='SDG-labels', ylabel='number of samples')

    axs[0, 0].set_xlabel('')
    axs[0, 1].set_ylabel('')
    axs[0, 1].set_xlabel('')
    axs[1, 1].set_ylabel('')

    # set the title of the entire figure
    fig.suptitle('SDG-label distribution in the datasets')

    # show the first figure
    plt.show()

    # concatenate the resulting dataframes
    df = pd.concat(dfs)

    # drop duplicates from the dataframe according to the text column
    df = df.drop_duplicates(subset=['text'])

    # now plot the label distribution for the merged dataframe

    # visualize the label distribution as a bar plot
    # sorted by the label index
    # y-axis is the number of samples
    # x-axis is the label index
    df['label'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Label Index")
    plt.ylabel("Number of Samples")
    plt.title("All Datasets Merged")
    plt.show()
    # create the kfold stratified splits
    # save the data to the directory as csv files for each fold
    # as train.csv, dev.csv and test.csv
    create_kfold_stratified_splits(df, "Data_merged_all")


if __name__ == '__main__':

    # list of datasets for training and testing
    datasets = ["original_data.csv",
                "politics.csv",
                "targets.csv",
                "osdg_v2023.csv"]

    # check if the Data folder exists
    # create it if it does not exist
    if not os.path.exists("./Data/"):
        os.makedirs("Data")

    # if the Data folder exists, check if the following datasets exist
    # if they do not exist, download them from the repository
    for file in datasets:
        if not os.path.exists("./Data/" + file):
            print("Downloading " + file + "...")
            hf_hub_download(repo_id="amay01/Quality_and_usability_SDG_dataset", repo_type="dataset", filename=file,
                            local_dir="./Data/")

    # At this point the data directory should be created and contain the datasets
    # we can now proceed with the data analysis and preprocessing
    main(datasets)
