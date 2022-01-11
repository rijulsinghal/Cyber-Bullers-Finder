#Import Commands
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from sklearn import svm
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import nest_asyncio
import tweepy
import instagram_private_api
import os
import pickle
from instagram_private_api import Client
nest_asyncio.apply()

#Train System

#Final Result found is stored in this dict
user_details_twitter = {}
user_details_instagram = {}

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, set):
          return list(obj)

       return json.JSONEncoder.default(self, obj)

def fitvectorizer(filename,vectorizer):
    df = pd.read_csv(filename)
    
    sent = []
    tokenize_sent = []
    clean_sent = []
    sent_str = ''

    stop_words = set(stopwords.words('english'))
    j = 0


    for i in df.comments:
        sent.append(i)
    
    for i in df.comments:
        tokenize_sent.append(word_tokenize(i.lower()))

    for w in tokenize_sent:
        for i in w:
            if i in stop_words:
                tokenize_sent[j].remove(i)
        j = j+1

    for i in tokenize_sent:
        for j in i:
            sent_str = sent_str + ' ' + j
        clean_sent.append(sent_str)
        sent_str = ''

    sentence_vectors = vectorizer.fit_transform(sent)

    return sentence_vectors

def trainmodels(sentence_vectors,filename):
    df = pd.read_csv(filename)
    
    X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, df.tagging, test_size=0.1)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    pickle.dump(clf,open('model.sav' , 'wb'));
    y_pred = clf.predict(X_test)

    print("Model1 Accuracy:",metrics.accuracy_score(y_test, y_pred))

    clf2 = LogisticRegression(solver='liblinear', C=1)
    clf2.fit(X_train, y_train)
    pickle.dump(clf2 , open('model_lr.sav' , 'wb'));
    y_pred2 = clf2.predict(X_test)

    print("Model2 Accuracy: ",metrics.accuracy_score(y_test, y_pred2))

    # print(sentence_vectors.shape);

def loadmodels():

    clf =  pickle.load(open('model.sav', 'rb'))
    clf2 =  pickle.load(open('model_lr.sav', 'rb'))

    # print("Models loaded successfully!!")
    return [clf,clf2]

def TwitterAuth(consumer_key,consumer_secret_key,access_token,access_secret_token):
    #Connecting to Twitter 
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret_key);
    auth.set_access_token(access_token,access_secret_token);
    api = tweepy.API(auth)
    return api

def fetchtweets(api,name,Limit):


    l = ['coordinates', 'text', 'geo', 'lang', 'place', 'user_name', 'user_screen_name', 'user_location', 'user_description', 'user_url', 'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_created_at', 'user_verified', 'user_profile_background_image_url' , 'user_profile_background_image_url_https' , 'user_profile_background_tile' , 'user_profile_image_url' , 'user_profile_image_url_https']
    res = {}
    for i in l:
        res[i] = []
    

    for tuit in tweepy.Cursor(api.search_tweets,q=name).items(Limit):
        res['coordinates'].append(tuit.coordinates)
        res['text'].append(tuit.text)
        res['geo'].append(tuit.geo)
        res['lang'].append(tuit.lang)
        res['place'].append(tuit.place)
        res['user_name'].append(tuit.user.name)
        res['user_screen_name'].append(tuit.user.screen_name)
        res['user_location'].append(tuit.user.location)
        res['user_description'].append(tuit.user.description)
        res['user_url'].append(tuit.user.url)
        res['user_followers_count'].append(tuit.user.followers_count)
        res['user_friends_count'].append(tuit.user.friends_count)
        res['user_listed_count'].append(tuit.user.listed_count)
        res['user_created_at'].append(tuit.user.created_at)
        res['user_verified'].append(tuit.user.verified)
        res['user_profile_background_image_url'].append(tuit.user.profile_background_image_url)
        res['user_profile_background_image_url_https'].append(tuit.user.profile_background_image_url_https)
        res['user_profile_background_tile'].append(tuit.user.profile_background_tile)
        res['user_profile_image_url'].append(tuit.user.profile_image_url)
        res['user_profile_image_url_https'].append(tuit.user.profile_image_url_https)

    df1 = pd.DataFrame(res)

    len = df1['text'].size

    #Deleting tweets other than english

    for i in range(0 , len):
        if(df1['lang'][i] != 'en'):
            df1.drop(labels=i , axis=0 , inplace = True)

    df1.to_csv('random_tweets.csv')  #Saving Tweets to CSV File
    random_tweets_list = df1['text'].values.tolist()

    return random_tweets_list

def cleantweets(random_tweets):
    for i in range(0,len(random_tweets)):
        random_tweets[i] = re.sub(r'http\S+', '', random_tweets[i])
        random_tweets[i] = re.sub(r'@\S+', '', random_tweets[i])

    return random_tweets

def vectorizetweets(predict_sent,vectorizer):

    sentence_vectors = vectorizer.transform(predict_sent)
    # print(sentence_vectors.shape)

    return sentence_vectors

def predictor(sentence_vectors, model):
    y_predict = model.predict(sentence_vectors)
    return y_predict

def suspected_user_details(predicted_list_model1,predicted_list_model2):
    df = pd.read_csv('random_tweets.csv')
    user_name = df['user_screen_name'].values.tolist()
    suspected_user_name = set()

    for i in range(len(predicted_list_model1)):
        if(predicted_list_model1[i] == 1):
            suspected_user_name.add(user_name[i])
    
    
    for i in range(len(predicted_list_model2)):
        if(predicted_list_model2[i] == 1):
            suspected_user_name.add(user_name[i])

    return suspected_user_name
    
def save_suspected_name(suspected_user_name):

    file = open("suspected_user_name.txt" , "a");
    df = pd.read_csv('random_tweets.csv')

    for i in suspected_user_name:
        file.write(i+"\n")

    file.close()


    suspected_df = df[df['user_screen_name'].isin(suspected_user_name)]
    global user_details_twitter
    user_details_twitter = suspected_df.to_dict('records')
    
def instagram(user_name , password , suspected_user_name):

    user_details = {}

    for i in suspected_user_name:
        try:
                api = Client(user_name, password)
                user_info = api.username_info(i)
                user_info['PersonalData'] = user_info.pop('user')


                followers_result = api.user_followers(user_info['PersonalData']['pk'] , instagram_private_api.Client.generate_uuid());
                followers_result['followers_data'] = followers_result.pop('users')
                
                following_result = api.user_following(user_info['PersonalData']['pk'] , instagram_private_api.Client.generate_uuid())
                following_result['following_data'] = following_result.pop('users')
                
                user_details[i] = [user_info,followers_result,following_result]

        except Exception as e:
                print(e)
                pass


    # print(user_details)
    
    global user_details_instagram
    user_details_instagram = user_details

def deletefiles():
    if os.path.exists("random_tweets.csv"):
        os.remove("random_tweets.csv")

    
    if os.path.exists("suspected_user_name.txt"):
        os.remove("suspected_user_name.txt")
    
def find_suspected_user(Query):

    user_details = {}

    vectorizer = TfidfVectorizer() # Vectorizer
    sentence_vectors = fitvectorizer('tweets.csv',vectorizer)

    # trainmodels(sentence_vectors,'tweets.csv') #Uncomment if new data is there to train

    # Loading Models
    model_list = loadmodels()
    clf = model_list[0]
    clf2 = model_list[1]

    #Twitter Authentication
    api = TwitterAuth("b8KRnyjlhemPxKq12vyZOWAaq","P5GZSBhLiOt9VXHZ9exo2uCtKj1WRp6AvYIT92IGHB8Hh5LK76","1448508667154731009-mqimm7eRTWKaDBxUvRgru67fxouWKk","GpmK3K8onr0B8aigDMWzHq8m68jf1kc68RasqeTGnfqn5")

    #No. of tweets required
    Limit = 1000

    random_tweets = fetchtweets(api,Query,Limit)
    cleaned_tweets = cleantweets(random_tweets)

    sentence_vectors = vectorizetweets(cleaned_tweets,vectorizer)
    predicted_value_model_1 = predictor(sentence_vectors,clf)
    predicted_value_model_2 = predictor(sentence_vectors,clf2)

    suspected_user_name = suspected_user_details(predicted_value_model_1,predicted_value_model_2)

    # print(suspected_user_name)

    save_suspected_name(suspected_user_name)  # Save Twitter Data

    instagram("hackingchunk","bullers@1234",suspected_user_name)  # Save Instagram Data

    user_details['suspected_user_id'] = suspected_user_name
    user_details['twitter'] = user_details_twitter
    user_details['instagram'] = user_details_instagram

    deletefiles() # Delete Temporary Files

    user_details = json.dumps(user_details,cls = SetEncoder)
    return user_details
    
# Taking Input from User
Query = input("Enter User-ID: ")
try:
    user_details = find_suspected_user(Query)
    print(user_details)
except :
    deletefiles()
    print("Something went wrong")

