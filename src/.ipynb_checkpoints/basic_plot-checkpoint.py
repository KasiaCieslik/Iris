from visualization.plot import polarity_distribution, target_distribution, word_cloud
import pandas as pd

df_balanced = pd.read_csv('../data/processed/prepared_df.csv')# after all preprocessing steps
df_unbalanced = pd.read_csv('../data/processed/unbalanced_df.csv')#before downsampling, with target
df_unbalanced_polarity = pd.read_csv('../data/processed/sum_of_polarity_df.csv')#before downsampling, with withou target

polarity_distribution(df_unbalanced_polarity, 'polarity_distribution_UNbalanced_dataset_without_threshold.png','Polarity distribution in UNbalanced dataset without threshold')
polarity_distribution(df_balanced, 'polarity_distribution_balanced_dataset_with_treshold.png','Polarity distribution in balanced dataset after setting threshold')

target_distribution(df_unbalanced,'target_unbalanced_dataset.png','Target distribution - Unbalanced dataset')
target_distribution(df_balanced,'target_balanced_dataset.png','Target distribution - Balanced dataset')

word_cloud(df_balanced,'positive')
word_cloud(df_balanced,'negative')