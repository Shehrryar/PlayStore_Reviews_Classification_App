import tkinter as tk
from UserInput import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from tkinter import messagebox as msg
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

class beam(tk.Tk):
    def __init__(self,parent):

        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.grid()

        self.ent1 = tk.Entry(self,width=50,bg="silver")#.place(x=140,y=80)
        self.ent1.grid(row=4,column=5)



        self.ent2 = tk.Entry(self,width=50,bg="silver")#.place(x=140,y=100)
        self.ent2.grid(row=5,column=5)
        self.ent3 = tk.Entry(self,width=50,bg="silver")#.place(x=140,y=120)
        self.ent3.grid(row=6,column=5)
        self.ent4 = tk.Entry(self,width=50,bg="silver")#.place(x=140,y=140)
        self.ent4.grid(row=7,column=5)


        # cls_btn = tk.Button(self,text='close', command=self.cls_win)
        # cls_btn.grid(row = 30, column = 2)

        #labek1 = tk.Label(self, text="Enter Review Information ", font=("arial", 15, "bold"), fg="black").place(x=50, y=30)
        lbE = tk.Label(self,text='Rating:',fg="Darkblue")#.place(x=40,y=80)
        lbE.grid(row=4,column=4,sticky='w')

        # lbE33 = tk.Label(self, text='')  # .place(x=40,y=80)
        # lbE33.grid(row=2, column=4)

        lbD = tk.Label(self,text='Review text:',fg="Darkblue")#.place(x=40,y=100)
        lbD.grid(row=5,column=4,sticky='w')
        lbd = tk.Label(self,text='App Title:',fg="Darkblue")#.place(x=40,y=120)

        lbd.grid(row=6,column=4,sticky='w')
        lbL = tk.Label(self,text='App Description:',fg="Darkblue")#.place(x=40,y=140)
        lbL.grid(row=7,column=4,sticky='w')


        lbL2 = tk.Label(self, text='Enter your review information',font=("arial",10,"bold"),fg="blue")  # .place(x=40,y=140)
        lbL2.grid(row=1, column=5)


        tk.Button(self, text="LR",  bg="lightblue",width=20, command=self.predict_logistic_regression).place(x=30,y=270)#.grid(row=9, column=2)
        tk.Button(self, text="SVM",  bg="lightblue",width=20, command=self.predict_SVM).place(x=300,y=270)#.grid(row=9, column=3)
        tk.Button(self, text="RF",  bg="lightblue",width=20,command=self.predict_Random_Forest).place(x=30,y=320)#.grid(row=9, column=4)
        tk.Button(self, text="NB",  bg="lightblue",width=20,command=self.predict_Naive_Bayes).place(x=300,y=320)#.grid(row=9, column=5)
        self.classifier_label = tk.Label(self, font=('Arial', 14, 'underline'))
        self.classifier_label.grid(columnspan=6, row=40, padx=5, pady=5)
        self.message_label = tk.Label(self, font=('Arial', 12, 'underline'))
        self.message_label.grid(columnspan=6, row=43, padx=5, pady=5)


    def getData(self):
        data_path = 'dataset/_70pecentage.csv'
        reviews_data = pd.read_csv(data_path)
        target_class = reviews_data.class_label
        reviews_data.drop(['review_id'], axis=1, inplace=True)
        reviews_data.drop(['app_id'], axis=1, inplace=True)
        reviews_data.drop(['class_label'], axis=1, inplace=True)

        return reviews_data, target_class
    def predict_logistic_regression(self):
        print("**LR**")
        #print(self.ent4.get())
        rating = self.ent1.get()
        review_text = self.ent2.get()
        app_title = self.ent3.get()
        app_description = self.ent4.get()
        feature = compute_sentiment_score(rating, review_text, app_title, app_description)
        #print(feature.head())
        train_X, train_y=self.getData()
        logreg = LogisticRegression()
        logreg.fit(train_X, train_y)
        y_pred = logreg.predict(feature)
        print("Output:",y_pred)
        result_label = ''
        if y_pred[0] == 0:
            # msg.showinfo("Prediction", "Reviews Not Spam")
            result_label = "Not-Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        if y_pred[0] == 1:
            # msg.showinfo("Prediction", "Spam Review")
            result_label = "Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        self.classifier_label.config(text="Logistic Regression Classification Model")
        self.message_label.config(text=result_label)

    def predict_SVM(self):
        print("**SVM**")
        #print(self.ent4.get())
        rating = self.ent1.get()
        review_text = self.ent2.get()
        app_title = self.ent3.get()
        app_description = self.ent4.get()
        feature = compute_sentiment_score(rating, review_text, app_title, app_description)
        #print(feature.head())
        train_X, train_y=self.getData()
        svm_classifier = svm.LinearSVC()
        svm_classifier.fit(train_X, train_y)
        y_pred = svm_classifier.predict(feature)
        print("Output:",y_pred)
        result_label = ''
        if y_pred[0] == 0:
            # msg.showinfo("Prediction", "Reviews Not Spam")
            result_label = "Not-Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        if y_pred[0] == 1:
            # msg.showinfo("Prediction", "Spam Review")
            result_label = "Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        self.classifier_label.config(text="Suport Vector Machine Classification Model")
        self.message_label.config(text=result_label)
    def predict_Random_Forest(self):
        print("**LR**")
        #print(self.ent4.get())
        rating = self.ent1.get()
        review_text = self.ent2.get()
        app_title = self.ent3.get()
        app_description = self.ent4.get()
        feature = compute_sentiment_score(rating, review_text, app_title, app_description)
        #print(feature.head())
        train_X, train_y=self.getData()
        randomforest_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        randomforest_classifier.fit(train_X, train_y)
        y_pred = randomforest_classifier.predict(feature)
        print("Output:",y_pred)
        result_label = ''
        if y_pred[0] == 0:
            # msg.showinfo("Prediction", "Reviews Not Spam")
            result_label = "Not-Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        if y_pred[0] == 1:
            # msg.showinfo("Prediction", "Spam Review")
            result_label = "Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        self.classifier_label.config(text="Random Forest Classification Model")
        self.message_label.config(text=result_label)
    def predict_Naive_Bayes(self):
        print("**Naive Bayes**")
        #print(self.ent4.get())
        rating = self.ent1.get()
        review_text = self.ent2.get()
        app_title = self.ent3.get()
        app_description = self.ent4.get()
        feature = compute_sentiment_score(rating, review_text, app_title, app_description)
        #print(feature.head())
        train_X, train_y=self.getData()
        navieBaysain_classifier= GaussianNB()
        navieBaysain_classifier.fit(train_X, train_y)
        y_pred = navieBaysain_classifier.predict(feature)
        print("Output:",y_pred)
        result_label=''
        if y_pred[0]==0:
            #msg.showinfo("Prediction", "Reviews Not Spam")
            result_label="Not-Spam Review: Predicted Label("+str(y_pred[0])+")"
        if y_pred[0]==1:
            #msg.showinfo("Prediction", "Spam Review")
            result_label = "Spam Review: Predicted Label(" + str(y_pred[0]) + ")"
        self.classifier_label.config(text="Navie Baysain Classification Model")
        self.message_label.config(text=result_label)


def main():

    app = beam(None)
    #w, h = app.winfo_screenwidth(), app.winfo_screenheight()
    app.geometry("%dx%d+%d+%d" % (480,400, int(app.winfo_screenwidth() / 2 - 500 / 2), int(app.winfo_screenheight() / 2 - 500 / 2)))
    app.title('Input Form')

    
    app.mainloop()

if __name__ == '__main__':
    main()