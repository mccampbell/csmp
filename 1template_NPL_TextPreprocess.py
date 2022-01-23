# -*- coding: utf-8 -*-
##########################################################################
# Name: Michael Campbell
# Date:
    from datetime import date
    today = date.today()
    print("Today's date:", today)
# Program: Python - 1template_DP_analysis-NPL.py
# Purpose: Natural Language Processing - allows machines to break down and understand human language
# Steps: 1. Import Libraries
#        2. Import Dataset
#        3. Text Preprocessing
# After text is obtained, Start with text normalization. Text normalization includes:
# A. Converting all letters to lower or upper case
# B. Converting numbers into words or removing numbers
# C. Removing punctuations, accent marks and other diacritics
# D. Removing white spaces
# E. Expanding abbreviations
# F. Removing stop words, sparse terms, and particular words
# G. Text canonicalization
##########################################################################
#
## 1. Import Libraries
import pandas as pd
import os                              # Operating system dependent functionality
import re
import glob
import seaborn as sns
## If you need to reinstall numpy
## !pip uninstall numpy
## !pip install numpy
###
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')
import matplotlib               as mpl # Additional plotting functionality
mpl.rcParams['figure.dpi'] = 400       # High resolution figures
import matplotlib.pyplot        as plt # Plotting package
import string
import nltk
from nltk.stem import PorterStemmer
import nltk.corpus
#
## nltk.download()  # Downloads all NLTK packages
### nltk.download('stopwords')
### nltk.download('punkt')
# Feature Selection - Identify strongest relationship with the output variable.  
# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#
# load data
os.chdir('U:\MikeC\Data')                         # Location of data files
## df = pd.read_csv('1analysis_dummy.csv')           # -*- coding: utf-8 -*-
#
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(filename, names=names)
#######################  Another way to read multiple files 8-18-21
# ### Location of files
for dirname, _, filenames in os.walk('/directory'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Feature Selection Identify relationship with the output variable -- might not need this section
from pandas inport read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
##
# Load Data
os.chdir('/directory')
# Load several .csv files
file_extension = ".csv"
all_filenames = [ i for i in glob.glob(f"file*{file_extension}") ]
print(all_filenames)
## You can create keys to partition the concatenated dataframes as each file is read in
df = pd.concat( [ read_csv(file) for file in all_filenames ], axis=0, keys=["key-label1","key-label2"] )
df.loc["key-label1"].count()
df.loc["key-label2"].count()
df.nunique()
#
## Output combined CSV just in case you need it
from datetime import datetime
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
print(timestr)
file_name = "/directory/filename"+timestr+".csv"
df.to_csv(file_name, index=False)
###
#################### 3. Texts Preprocessing ################# 
# A. Convert Text to Lower Case
texts=["CANADA","Canada","canadA","canada"]
lower_words=[word.lower() for word in texts]
lower_words
#
# B. Converting numbers into words or removing numbers
input_str = 'Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.'
result = re.sub(r'\d+', '', input_str)
print(result)
#
# C. Removing punctuations, accent marks and other diacritics
input_str = "This &is [an] example? {of} string. with.? punctuation!!!!"       # Sample string
result = input_str.translate(string.maketrans("",""), string.punctuation)
print(result)
#
# D. Removing white spaces
input_str = " \t a string example\t "
input_str = input_str.strip()
input_str
# E. Expanding abbreviations
# F. Removing stop words, sparse terms, and particular words
## “Stop words” are the most common words in a language like “the”, “a”, “on”, “is”, “all”.
## Check the list of stopwords
import nltk
### nltk.download('stopwords')
### nltk.download('punkt')
from nltk.corpus import stopwords
print(stopwords.words('english'))

stopwords=['this','that','and','a','we','it','to','is','of','up','need']
text="this is a text full of content and we need to clean it up"
words=text.split(" ")
shortlisted_words=[]

# Check if NLTK files are installed. For example, WORDNET #################################
print (os.listdir(nltk.data.find("corpora"))) # Add data formats downloaded from nltk.download()  # Downloads all NLTK packages
dir(nltk.corpus)
nltk.find('corpora/wordnet')
nltk.corpus.gutenberg.fileids()
## Example, look at the file shakespeare-hamlet.txt
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet
## Say, I want to look at the first 500 elements in hamlet
for word in hamlet[:500]:
    print(word, sep = ' ', end = ' ')
###########################################################################################
## If installed my own corpus in the nltk_data/corpora area and the NLTK does not know about it, Fire up the appropriate reader yourself.
##  E.g., if it is a plaintext corpus in corpora/mycorpus and all the files end in .txt, do it like this:

# import nltk
# from nltk.corpus import PlaintextCorpusReader

# mypath = nltk.data.find("corpora/mycorpus")
# mycorpus = PlaintextCorpusReader(mypath, r".*\.txt$")
# But, in that case I could put my own corpus anywhere, and point mypath to it directly instead of asking the NLTK to find it.
############################################################################################
#remove stop words
for w in words:
    if w not in stopwords:
        shortlisted_words.append(w)
    else:
        shortlisted_words.append("W")

print("original sentence = ",text)    
print("sentence with stop words removed= ",' '.join(shortlisted_words))

# OR
input_str = "NLTK is a leading platform for building Python programs to work with human language data."
stop_words = set(stopwords.words("english"))
from nltk.tokenize import word_tokenize
tokens = word_tokenize(input_str)
result = [i for i in tokens if not i in stop_words]
print (result)

### scikit-learn tool also provides a stop words list
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
##
# Also possible to use spaCy, a free open-source library
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Stemming is a process of reducing words to their word stem, base or root
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()
input_str='There are several types of stemming algorithms.'
input_str=word_tokenize(input_str)
for word in input_str:
    print(stemmer.stem(word))
### Another way:
# init stemmer
porter_stemmer=PorterStemmer()
words=["connect","connected","connection","connections","connects"]
stemmed_words=[porter_stemmer.stem(word=word) for word in words]

stemdf= pd.DataFrame({'original_word': words,'stemmed_word': stemmed_words})
stemdf

# 
## Lemmatization
# The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form. 
# As opposed to stemming, lemmatization does not simply chop off inflections. Instead it uses lexical knowledge bases to get the correct base forms of words.
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer=WordNetLemmatizer()
input_str='been had done languages cities mice'
input_str=word_tokenize(input_str)
for word in input_str:
    print(lemmatizer.lemmatize(word))
    
## Or
#lemmatize trouble variations
words=["trouble","troubling","troubled","troubles",]
lemmatized_words=[lemmatizer.lemmatize(word=word,pos='v') for word in words]
lemmatizeddf= pd.DataFrame({'original_word': words,'lemmatized_word': lemmatized_words})
lemmatizeddf=lemmatizeddf[['original_word','lemmatized_word']]
lemmatizeddf   
#############
## Part of speech tagging (POS) using TextBlob
input_str='Parts of speech examples: an article, to write, interesting, easily, and, of'
from textblob import TextBlob
result = TextBlob(input_str)
print(result.tags)

#
# Chunking is a natural language process that identifies constituent parts of sentences (nouns, verbs, adjectives, etc.) 
# and links them to higher order units that have discrete grammatical meanings (noun groups or phrases, verb groups, etc.)

## The first step is to determine the part of speech for each word:
REACH 

# G. Text canonicalization

## Tokenization: The process of splitting the given text into smaller pieces called tokens.

################################ End Text Preprocessing ###########################################
list(df['col1'])           # Display the content of a column
df.columns.values.tolist() # List all columns      
#########################
array = df.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
#
# Describe the data and check the column (e.g Error) that is the reason for why I want to predict
# 1. Data Exploration
#
df.columns 
df.info()                          # Obtain the data type of the columns 
df.head()
df.shape                           # Total records; Verify that the number of unique versus
                                   # total records are not the same. This tells whether the
                                   # unique column identified is really the Primary Key
                                   # for the record.
df['host'].nunique(dropna=False)   # Counts unique values
id_counts = df['host'].value_counts() # List Primary Keys and how often they occur
id_counts.head()
id_counts.value_counts()              # Number of grouped duplicated entries
df['severity_id_and_name'].describe()
#
### Use Boolean Mask to filter an array, or series by some condition
np.random.seed(seed=24)
random_integers=np.random.randint(low=1,high=5,size=100)
random_integers[0:5]                  # First 5 elements for random numbers
is_equal_to_3 = random_integers==3    # Locations of all elements of random_integers equal to 3
dupe_mask = id_counts==2              # Locates the duplicates
id_counts.index[0:5]                  # Displays the first 5 rows of the index
dupe_ids = id_counts.index[dupe_mask] # Stores the duplicates PK in a new variable: dupe_ids
## Convert the dupe_ids to a list and obtain the length
dupe_ids = list(dupe_ids)
len(dupe_ids)
dupe_ids[0:5]
## Identify if duplicates have 1 row with data and another with just 0s -- Delete the zero rows 
df.loc[df ['ID'].isin(dupe_ids[0:3]), : ].head(10)
#################
######## Data Clean
#
# Data Cleaning includes identify missing and redundant data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#
##### Deal with missing data
df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
df = df.drop(df.loc[df['message_txt'].isnull()].index)
df.isnull().sum().max() # Checking that there is no more missing data
#
# Find rows that have all 0s starting with the second row
# 1. Create a boolean matrix
df_zero_mask = df == 0
# 2. Create the boolean series to identify every row where all elements starting from the 2nd column are all zero.
feature_zero_mask = df_zero_mask.iloc[: , 1:].all(axis=1)
## The output number tells the numhber of rows for every column except the first column
sum(feature_zero_mask)
# 3. Clean the dataframe by eliminating the rows with all 0s, except the PK
df_clean_1 = df.loc[~feature_zero_mask, : ].copy[]  # Copy into new dataframe (df_clean_1)
df_clean_1.shape              #  Verify the number of rows in df_clean_1
df_clean_1['PK'].nunique      # Gets the number of unique IDs
df_clean_1.info()             # Identify the data columns for values that are the feature
df_clean_1['column'].head(5) # Views the first 5 rows
df_clean_1['column'].value_counts() # Obtain value counts
value_pay_1_mask = df_clean_1['PAY_1'] != 'Not Available' # Find rows that do not have missing data
sum(valid_pay_1_mask)   # Check how many rows that have no missing data by doing a SUM
df_clean_2 = df_clean_1.loc[valid_pay_1_mask, :].copy() # Eliminates rows with missing data
df_clean_2.shape
df_clean_2['PAY_1].value_counts()
# Cast from the generic object type to int64 using the .astype method
df_clean_2['PAY_1'] = df_clean_2['PAY_1'].astype('int64')
df_clean_2[['PAY_1', 'PAY_2']].info()
##########
#
##### Univariate Analysis - Standardizing data
# Here, the primary concern is to establish a threshold that defines an observation as an 
# outlier. To do so, standardize the data. Data standardization means converting data 
# values to have mean of 0 and a standard deviation of 1.

error_scaled = StandardScaler().fit_transform(df['severity_id_and_name'][:,np.newaxis])
low_range = error_scaled[error_scaled[:,0].argsort()][:10]
high_range= error_scaled[error_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\outer range (high) of the distribution:')
print(high_range)
#
###### Bivariate Analysis 
# Outliers must be deleted.  Look out for trends.  For example, the two largest numbers 
# might be following a trend; so keep them.

var = 'GrLivArea'
data = pd.concat([df['severity_id_and_name'], df[var]], axis=1)
data.plot.scatter(x=var, y='severity_id_and_name', ylim=(0,800000))

# Delete Points
df.sort_values(by = 'Host', ascending = False)[:2]
df = df.drop(df[df['host'] == 1299].index)
df = df.drop(df[df['host'] == 524].index)

# Bivariate Analysis numberVariable/numberVariable
var = 'Totalhost'
data = pd.concat([df['host'], df[var]], axis=1)
data.plot.scatter(x=var, y='host', ylim=(0,800000))
#
# Histogram
df_clean_2[['LIMIT_BAL', 'AGE']].hist() # Histogram to visualize data that is on a continuos scale
df_clean_2[['LIMIT_BAL', 'AGE']].describe() # Tabular report of summary statistics
df_clean_2.groupby('Column').agg( { 'Power failure pred':'mean'}).plot.bar(legend=False)
plt.ylabel('Expect Failure')
plt.xlabel('Error Level: ordinal encoding')
#
sns.distplot(df['severity_id_and_name'])
#
#skewness and kurtosis
skew(df["Returns"].dropna())          # Drop any NA values
kurtosis(df["Returns"].dropna())
print("Skewness: %f" % df['severity_id_and_name'].skew())
print("Kurtosis: %f" % df['severity_id_and_name'].kurt())
#
# Scatterplot to visualize the data for conclusions
var = 'host'        # Has to be a column from the df that you want to show the relations
data = pd.concat([df['severity_id_and_name'], df[var]], axis=1)
data.plot.scatter(x=var, y='severity_id_and_name', ylim=(0,800000))
#
# Correlation matrix (Heap Style)
k = 10                       #number of variables for heatmap
cols = corrmat.nlargest(k, 'severity_id_and_name')['severity_id_and_name'].index
cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
               yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#
#### One Hot Encoding transforms categorical feature into numerical feature that can be used in mathematical models.
# 1. Create an empty column and examine the first few rows of the datafreame
df_clean_2['EDUCATION_CAT' ] = 'none'   ## This is t he new column created
df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10)
#
# 2. Create a dictionary that maps the educational(new column) categories
cat_mapping = {
    1: "graduate school",
    2: "university",
    3: "high school",
    4: "others"
}
#
# Apply mapping to the original EDUCATION column and assign result to the new EDUCATION_CAT column
df_clean_2['EDUCATION_CAT'] = df_clean_2['EDUCATION'].map(cat_mapping)
df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10)
#
# 3. Create a one hot encoded dataframe of the EDUCATION_CAT column.
## One Hot Encoded colums are referred to as dummy variables.
edu_ohe = pd.get_dummies(df_clean_2['EDUCATION_CAT'])
edu_ohe.head(10)
#
# 4. Concatenate the one-hot encoded DataFrame to the original DataFrame
df_with_ohe = pd.concat([df_clean_2, edu_ohe], axis=1)
df_with_ohe[['EDUCATION_CAT', 'graduate school',
             'high school', 'university', 'others']].head(10)
#
# 5. Write the latest DataFrame to a CSV file without the INDEX
df_with_ohe.to_csv('directory/filename.csv' , index=False)
