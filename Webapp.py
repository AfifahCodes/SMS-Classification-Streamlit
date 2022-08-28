import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk 
nltk.download('wordnet')
nltk.download("popular")
import spacy
import string
#Reading Data
df = pd.read_csv("SMSdata.csv")


#Cleaning Data

#Ensuring message_body is in string format
df["Message_body"] = df["Message_body"].astype(str)

#Lowecase all words
df["Message_body"]= df["Message_body"].str.lower()
df.head()

# #Remove the URLs & Numbers
df['NoURL'] = df['Message_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['NoNum'] = df['NoURL'].str.replace('\d+', '')

# Removal of Punctuation
df["No_Punc"] = df['NoNum'].str.replace('[^\w\s]','')
# Removal of Stop Words
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["No_stop"] = df["No_Punc"].apply(lambda text: remove_stopwords(text))
df.head()

#Removal of Frequent Words
from collections import Counter
cnt = Counter()
for text in df["No_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

df["No_FW"] = df["No_stop"].apply(lambda text: remove_freqwords(text))
df.head()

# Lemmatize the Words
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["text_lemmatized"] = df["No_FW"].apply(lambda text: lemmatize_words(text))


#Most Common words within spam
#Withn Spam
Spam = df[df['Label']== 'Spam'] 
Spam.head()
Spam_freq_words = Counter(" ".join(Spam["text_lemmatized"]).split()).most_common(20)



#Most Common words within spam
#Withn Non-Spam
NonSpam = df[df['Label']== 'Non-Spam'] 
NonSpam.head()
NonSpam_freq_words = Counter(" ".join(NonSpam["text_lemmatized"]).split()).most_common(20)



from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
text = Spam['text_lemmatized'].values 

wordcloud_spam = WordCloud().generate(str(text))

plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()


NonSpamtext = NonSpam['text_lemmatized'].values 

wordcloud_nonSpam = WordCloud().generate(str(NonSpamtext))

plt.imshow(wordcloud_nonSpam)
plt.axis("off")
plt.show()


def main():
    st.title('SMS Classification')
    #Obtain the year from data + drop duplicates (can also use drop.duplicates)
    Label = st.selectbox( 'Select Message type', df['Label'].drop_duplicates())
    filtered = df[df['Label']== Label ]
    
    button = st.button('Display data')
    if button:
        column1, column2 = st.columns(2)
        # date = df["Date_Received"]
        if Label == 'Spam':
            with column1:
                Spamtext = Spam['text_lemmatized'].values 
                wordcloud_Spam = WordCloud().generate(str(Spamtext))
                plt.imshow(wordcloud_Spam)
                plt.axis("off")
                st.pyplot(plt)
        else:
            with column1:
                NonSpamtext = NonSpam['text_lemmatized'].values 
                wordcloud_nonSpam = WordCloud().generate(str(NonSpamtext))
                plt.imshow(wordcloud_nonSpam)
                plt.axis("off")
                st.pyplot(plt)
            
        with column2:
            count = df.groupby('Date_Received')['Message_body'].count()
            sorted_count = count.sort_values(ascending = False)
            sorted_top = sorted_count.head(500)
            st.line_chart(sorted_top)

    #st.table(data =filtered)
if __name__ == "__main__":
    main()
