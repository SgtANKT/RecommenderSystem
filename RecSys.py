import pandas as pd
import numpy as np
import regex as re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
ps = PorterStemmer()
tfidf_vectorizer = TfidfVectorizer()


class RecSys:
    def __init__(self):
        pass

    def black_txt(self, token):
        return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2

    def clean_txt(self, text):
        stem_text = []
        lem_text = []
        clean_text = []
        text = re.sub("'", "", text)
        text = re.sub("(\\d|\\W)+", " ", text)
        text = text.replace("nbsp", "")
        lem_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if self.black_txt(word)]
        #     clean_text = [word for word in lem_text if black_txt(word)]
        stem_text = [ps.stem(word) for word in lem_text if self.black_txt(word)]
        return " ".join(stem_text)

    def split_diff(self, text):
        prefix_list = []
        spl = text.split()
        prefix_list.append(spl[0])
        return " ".join(prefix_list)


    def course_files(self):
        coursera_df = pd.read_csv(r"C:\Users\ankit\Desktop\Recommender\data\Courses_Combined_Coursera.csv")
        udemy_df = pd.read_csv(r"C:\Users\ankit\Desktop\Recommender\data\udemy_courses.csv")
        # drop nan
        coursera_df.drop(columns='CourseCurator', inplace=True)
        # add course id
        coursera_df['CourseId'] = np.random.randint(1000, 99999, size=coursera_df.shape[0])
        coursera_df['CourseId'] = coursera_df['CourseId'].astype('int64')

        # drop nan
        coursera_df.replace(to_replace ="NaN", value =np.nan)
        coursera_df.dropna(inplace=True)

        # changing column names
        coursera_df["Difficulty"] = coursera_df["Difficulty"].replace(to_replace ="Mixed", value ='Intermediate')
        coursera_df['Difficulty'].unique()


        # changing columns of course df from udemy
        udemy_df = udemy_df.rename(columns={'course_title': 'CourseName', 'level': 'Difficulty', 'num_reviews': 'TotalRatings', 'course_id': 'CourseId'})
        udemy_df.drop(columns=['url', 'is_paid', 'price', 'num_subscribers', 'num_lectures', 'content_duration', 'published_timestamp', 'subject'], inplace=True)

        # Adding avg ratings
        udemy_df['AverageRatings']=np.random.uniform(2.5,4.9,size=udemy_df.shape[0]).astype(float).round(1)

        #
        udemy_df['Difficulty'] = udemy_df['Difficulty'].apply(self.split_diff)
        # udemy_df['Difficulty'].unique()
        udemy_df["Difficulty"].replace(to_replace="All", value='Beginner', inplace=True)
        udemy_df["Difficulty"].replace(to_replace="Expert", value='Advanced', inplace=True)

        course_df = pd.concat([coursera_df, udemy_df], ignore_index=True)
        course_df = course_df[['CourseId', 'CourseName', 'Difficulty', 'AverageRatings', 'TotalRatings']]

        return course_df

    def vectorise_courses(self, course_df):

        tfidf_jobid = tfidf_vectorizer.fit_transform((course_df['CourseName']))  # fitting and transforming the vector
        return tfidf_jobid

    def experience(self):
        df_experience = pd.read_csv(r"C:\Users\ankit\Desktop\Recommender\data\Experience.csv")

        df_experience = df_experience[['Applicant.ID', 'Position.Name']]
        # cleaning the text
        df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(self.clean_txt)
        # df_experience['Position.Name'] = df_experience['Position.Name'].str.replace('[^a-zA-Z \n\.]',"")
        df_experience['Position.Name'] = df_experience['Position.Name'].str.lower()
        df_experience = df_experience.sort_values(by='Applicant.ID')
        df_experience = df_experience.fillna(" ")
        df_experience = df_experience.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()

        return df_experience

    def poi(self):
        df_poi = pd.read_csv(r"C:\Users\ankit\Desktop\Recommender\data\Positions_Of_Interest.csv", sep=',')
        df_poi = df_poi.drop('Updated.At', 1)
        df_poi = df_poi.drop('Created.At', 1)

        # cleaning the text
        df_poi['Position.Of.Interest'] = df_poi['Position.Of.Interest'].map(str).apply(self.clean_txt)
        # df_poi['Position.Of.Interest']=df_poi['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")

        df_poi = df_poi.fillna(" ")
        df_poi = df_poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

        return df_poi

    def merge_all(self):
        df_experience = self.experience()
        df_poi = self.poi()
        # Merge experience and position of interest
        df_exp_poi = df_experience.merge(df_poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_exp_poi = df_exp_poi.fillna(' ')
        df_exp_poi = df_exp_poi.sort_values(by='Applicant.ID')

        df_exp_poi["position_of_interest"] = df_exp_poi["Position.Name"].map(str) + " " + df_exp_poi["Position.Of.Interest"]
        df_pos_of_interest_person = df_exp_poi[['Applicant.ID', 'position_of_interest']]
        df_pos_of_interest_person['position_of_interest'] = df_pos_of_interest_person['position_of_interest'].apply(
            self.clean_txt)

        return df_pos_of_interest_person


    def get_recommendation(self, top, df_all, scores, user_id):
        # course_df = course_files()
        recommendation = pd.DataFrame(columns = ['ApplicantID','CourseId', 'CourseName'])[:10]
        count = 0
        for i in top:
            recommendation.at[count, 'ApplicantID'] = user_id
            recommendation.at[count, 'CourseId'] = df_all['CourseId'][i]
            recommendation.at[count, 'CourseName'] = df_all['CourseName'][i]
            # recommendation.at[count, 'Difficulty'] = df_all['Difficulty'][i]
            # recommendation.at[count, 'Score'] =  scores[count]
            count += 1
        return recommendation



    def main_dynamic(self, position):
        print('Performing Pre-requisits')
        course_df = self.course_files()
        df_pos_of_interest_person = self.merge_all()

        # position_interest = input('Enter your desired position: ')
        user_id = np.random.randint(420420, 999999)


        print("Vectorizing courses")
        tfidf_jobid = self.vectorise_courses(course_df=course_df)


        df_pos_of_interest_person.loc[len(df_pos_of_interest_person.index)] = [user_id, position]
        df_pos_of_interest_person['position_of_interest'] = df_pos_of_interest_person['position_of_interest'].apply(
            self.clean_txt)

        index = np.where(df_pos_of_interest_person['Applicant.ID'] == user_id)[0][0]
        user_q = df_pos_of_interest_person.iloc[[index]]


        # vectorize users preferences
        print("Vectorizing user preferences")
        user_tfidf = tfidf_vectorizer.transform(user_q['position_of_interest'])

        cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_jobid)

        # outputting vectorised values to a list
        output2 = list(cos_similarity_tfidf)
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        recommends = self.get_recommendation(top, course_df, list_scores, user_id)
        rec_json = recommends.to_json(f"recommender_json/recommends{user_id}.json")
        print("Json Created")
        return rec_json

    def main(self):
        print('Performing Pre-requisits')
        course_df = self.course_files()
        df_pos_of_interest_person = self.merge_all()

        position_interest = input('Enter your desired position: ')
        user_id = np.random.randint(420420, 999999)


        print("Vectorizing courses")
        tfidf_jobid = self.vectorise_courses(course_df=course_df)


        df_pos_of_interest_person.loc[len(df_pos_of_interest_person.index)] = [user_id, position_interest]
        df_pos_of_interest_person['position_of_interest'] = df_pos_of_interest_person['position_of_interest'].apply(
            self.clean_txt)

        index = np.where(df_pos_of_interest_person['Applicant.ID'] == user_id)[0][0]
        user_q = df_pos_of_interest_person.iloc[[index]]


        # vectorize users preferences
        print("Vectorizing user preferences")
        user_tfidf = tfidf_vectorizer.transform(user_q['position_of_interest'])

        cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_jobid)

        # outputting vectorised values to a list
        output2 = list(cos_similarity_tfidf)
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        recommends = self.get_recommendation(top, course_df, list_scores, user_id)
        # rec_json = recommends.to_json(f"recommender_json/recommends{user_id}.json")
        rec_json = recommends.to_dict(orient='records')
        print("Json Created")
        return rec_json


if __name__ == '__main__':

    rec = RecSys()
    print(rec.main())
