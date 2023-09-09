import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class Spam_Message_detector:
    def __init__(self):
    #loading data to pandas
        
        raw_mail_data= pd.read_csv('spam.csv')
        # print(raw_mail_data)

        mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

        #label ham=1 spam=0

        mail_data.loc[mail_data['v1']== 'spam','v1']= 0
        mail_data.loc[mail_data['v1']==  'ham','v1'] = 1


        x = mail_data['v2']
        y = mail_data['v1']

        #splitting data and testing

        x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.2 , random_state=3)
        # print(x_train)

        #USING FEATURE EXTRACTING TO NUMERICALIZE EACH MESSAGE
        self.feature_extr= TfidfVectorizer(min_df=1 , stop_words='english')
        x_train_feature = self.feature_extr.fit_transform(x_train)
        x_test_feature = self.feature_extr.transform(x_test) 


        #convert ytrain ytest as int

        y_train= y_train.astype('int')
        y_test= y_test.astype('int')

        #applying logistic regression

        self.model = LogisticRegression()
        #training logistic model with training data

        self.model= self.model.fit(x_train_feature, y_train)

        #evaluate model
        prediction_on_training =  self.model.predict(x_train_feature)
        self.accuracy_score_checking_training= accuracy_score(y_train, prediction_on_training)



        prediction_on_testing =  self.model.predict(x_test_feature)
        self.accuracy_score_checking_test= accuracy_score(y_test, prediction_on_testing)

        # print("training  ",self.accuracy_score_checking_training)
        # print("testing    ",self.accuracy_score_checking_test)
        # print(y_test,prediction_on_testing)

    #predictive system
    def NewPrediction(self,content):
        input_mail=[content]

        #convert text to feature vectors
        input_data_extraction= self.feature_extr.transform(input_mail)
        prediction= self.model.predict(input_data_extraction)
        results = {'predict':prediction[0], 'training': self.accuracy_score_checking_training, 'testing': self.accuracy_score_checking_test}
        return results



# content = "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, �1.50 to rcv"
# obj= Spam_Message_detector()
# print(obj.NewPrediction(content))