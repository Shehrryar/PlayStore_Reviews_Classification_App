from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

##########################################################################################################

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)
    plt.show()
def prepareReviewsData(reviews_data):
    target_class= reviews_data.class_label
    reviews_data.drop(['review_id'], axis=1, inplace=True)
    reviews_data.drop(['app_id'], axis=1, inplace=True)
    reviews_data.drop(['class_label'], axis=1, inplace=True)

    return reviews_data, target_class

def logisticRegression(reviews_data, flag):
    Result = []
    print(reviews_data.head())
    reviews_data, target_class= prepareReviewsData(reviews_data)
    X_train, X_test, y_train, y_test = train_test_split(reviews_data, target_class, test_size=0.3, random_state=14)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("***** Logistic Regression Classifier*****")
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    #Result.append("Accuracy Score" + str(round(logreg.score(X_test, y_test),3)))

    # Accuracy
    accuracy = round(accuracy_score(y_test, y_pred),3)
    print("Accuracy (Logistic Regression):", accuracy)
    Result.append("Accuracy (Logistic Regression)")
    Result.append("Accuracy (Logistic Regression):"+ str(accuracy))
    # F1 Score
    f_score = round(f1_score(y_test, y_pred), 3)
    print("F1 Score (Logistic Regression):", f_score)
    Result.append("F1 Score (Logistic Regression)")
    Result.append("F1 Score (Logistic Regression):"+ str(f_score))
    #Precision
    precision = round(precision_score(y_test, y_pred), 3)
    print("Precision  (Logistic Regression):", precision)
    Result.append("Precision (Logistic Regression)")
    Result.append("Precision (Logistic Regression):"+ str(precision))
    # Precision
    recall = round(recall_score(y_test, y_pred), 3)
    print("Recall (Logistic Regression):", recall)
    Result.append("Recall (Logistic Regression)")
    Result.append("Recall (Logistic Regression):"+ str(recall))
    #Confussion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    Result.append("Confusion Matrix (Logistic Regression)")
    Result.append("Confusion Matrix (Logistic Regression):"+ str(c_matrix))
    #Classification report
    c_report= classification_report(y_test, y_pred)
    print(c_report)
    Result.append("Classification Report (Logistic Regression)")
    Result.append("Classification Report (Logistic Regression):"+ str(c_report))
    #ROC Curve
    # logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #Resuts Visulization
    plt.style.use('ggplot')

    x = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    performance_score= [None] * 4
    performance_score[0]=accuracy * 100
    performance_score[1]=f_score * 100
    performance_score[2]= precision * 100
    performance_score[3]=recall *100

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance_score, color='#CD853F')
    plt.xlabel("Performance Measure")
    plt.ylabel("Score")
    plt.title("Performance Analysis of Logistic Regression Model")

    plt.xticks(x_pos, x)
    plt.savefig("LR_results")
    plt.show()

    return Result
def supportVectorMachine(reviews_data, flag):
    Result = []
    print(reviews_data.head())
    reviews_data, target_class = prepareReviewsData(reviews_data)
    X_train, X_test, y_train, y_test = train_test_split(reviews_data, target_class, test_size=0.3, random_state=14)
    svm_classifier = svm.LinearSVC()
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    print("***** Support Vector Machine Classifier*****")
    print('Accuracy of Support Vector Machine Classifier on test set: {:.2f}'.format(svm_classifier.score(X_test, y_test)))

    # Result.append("Accuracy Score" + str(round(logreg.score(X_test, y_test),3)))

    # Accuracy
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print("Accuracy (Support Vector Machine):", accuracy)
    Result.append("Accuracy (Support Vector Machine)")
    Result.append("Accuracy (Support Vector Machine):" + str(accuracy))
    # F1 Score
    f_score = round(f1_score(y_test, y_pred), 3)
    print("F1 Score (Support Vector Machine):", f_score)
    Result.append("F1 Score (Support Vector Machine)")
    Result.append("F1 Score (Support Vector Machine):" + str(f_score))
    # Precision
    precision = round(precision_score(y_test, y_pred), 3)
    print("Precision  (Support Vector Machine):", precision)
    Result.append("Precision (Support Vector Machine)")
    Result.append("Precision (Support Vector Machine):" + str(precision))
    # Precision
    recall = round(recall_score(y_test, y_pred), 3)
    print("Recall (Support Vector Machine):", recall)
    Result.append("Recall (Support Vector Machine)")
    Result.append("Recall (Support Vector Machine):" + str(recall))
    # Confussion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    Result.append("Confusion Matrix (Support Vector Machine)")
    Result.append("Confusion Matrix (Support Vector Machine):" + str(c_matrix))
    # Classification report
    c_report = classification_report(y_test, y_pred)
    print(c_report)
    Result.append("Classification Report (Support Vector Machine)")
    Result.append("Classification Report (Support Vector Machine):" + str(c_report))
    # ROC Curve
    # logit_roc_auc = roc_auc_score(y_test, svm_classifier.predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, svm_classifier.predict(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Support Vector Machine (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    # Resuts Visulization
    plt.style.use('ggplot')

    x = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    performance_score = [None] * 4
    performance_score[0] = accuracy * 100
    performance_score[1] = f_score * 100
    performance_score[2] = precision * 100
    performance_score[3] = recall * 100

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance_score, color='#D68910')
    plt.xlabel("Performance Measure")
    plt.ylabel("Score")
    plt.title("Performance Analysis of Support Vector Machine Model")

    plt.xticks(x_pos, x)
    plt.savefig("SVM_results")
    plt.show()

    return Result
def randomForest(reviews_data, flag) :
    Result = []
    print(reviews_data.head())
    reviews_data, target_class = prepareReviewsData(reviews_data)
    X_train, X_test, y_train, y_test = train_test_split(reviews_data, target_class, test_size=0.3, random_state=14)
    randomforest_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    randomforest_classifier.fit(X_train, y_train)
    y_pred = randomforest_classifier.predict(X_test)
    print("***** Random Forest Classifier*****")
    print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(
        randomforest_classifier.score(X_test, y_test)))

    # Result.append("Accuracy Score" + str(round(logreg.score(X_test, y_test),3)))

    # Accuracy
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print("Accuracy (Random Forest):", accuracy)
    Result.append("Accuracy (Random Forest)")
    Result.append("Accuracy (Random Forest):" + str(accuracy))
    # F1 Score
    f_score = round(f1_score(y_test, y_pred), 3)
    print("F1 Score (Random Forest):", f_score)
    Result.append("F1 Score (Random Forest)")
    Result.append("F1 Score (Random Forest):" + str(f_score))
    # Precision
    precision = round(precision_score(y_test, y_pred), 3)
    print("Precision  (Random Forest):", precision)
    Result.append("Precision (Random Forest)")
    Result.append("Precision (Random Forest):" + str(precision))
    # Precision
    recall = round(recall_score(y_test, y_pred), 3)
    print("Recall (Random Forest):", recall)
    Result.append("Recall (Random Forest)")
    Result.append("Recall (Random Forest):" + str(recall))
    # Confussion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    Result.append("Confusion Matrix (Random Forest)")
    Result.append("Confusion Matrix (Random Forest):" + str(c_matrix))
    # Classification report
    c_report = classification_report(y_test, y_pred)
    print(c_report)
    Result.append("Classification Report (Random Forest)")
    Result.append("Classification Report (Random Forest):" + str(c_report))
    #ROC Curve
    # logit_roc_auc = roc_auc_score(y_test, randomforest_classifier.predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, randomforest_classifier.predict(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Support Vector Machine (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    plt.style.use('ggplot')

    x = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    performance_score = [None] * 4
    performance_score[0] = accuracy * 100
    performance_score[1] = f_score * 100
    performance_score[2] = precision * 100
    performance_score[3] = recall * 100

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance_score, color='#2471A3')
    plt.xlabel("Performance Measure")
    plt.ylabel("Score")
    plt.title("Performance Analysis of Random Forest Model")

    plt.xticks(x_pos, x)
    plt.savefig("RF_results")
    plt.show()

    #Features Importanct
    # features_importance= randomforest_classifier.feature_importances_
    # for i, v in enumerate(features_importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # # plot feature importance
    # plt.bar([x for x in range(len(features_importance))], features_importance)
    # plt.show()
    base_imp = imp_df(X_train.columns, randomforest_classifier.feature_importances_)
    print(base_imp)
    #var_imp_plot(base_imp, 'Default feature importance (RF)')
    return Result
def navieBaysain(reviews_data, flag):
    Result = []
    print(reviews_data.head())
    reviews_data, target_class = prepareReviewsData(reviews_data)
    X_train, X_test, y_train, y_test = train_test_split(reviews_data, target_class, test_size=0.3, random_state=14)
    navieBaysain_classifier= GaussianNB()
    navieBaysain_classifier.fit(X_train, y_train)
    y_pred= navieBaysain_classifier.predict(X_test)
    print("***** Navie Baysain Classifier*****")
    print('Accuracy of Navie Baysain Classifier on test set: {:.2f}'.format(
        navieBaysain_classifier.score(X_test, y_test)))

    # Result.append("Accuracy Score" + str(round(logreg.score(X_test, y_test),3)))

    # Accuracy
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print("Accuracy (Navie Baysain):", accuracy)
    Result.append("Accuracy (Navie Baysain)")
    Result.append("Accuracy (Navie Baysain):" + str(accuracy))
    # F1 Score
    f_score = round(f1_score(y_test, y_pred), 3)
    print("F1 Score (Navie Baysain):", f_score)
    Result.append("F1 Score (Navie Baysain)")
    Result.append("F1 Score (Navie Baysain):" + str(f_score))
    # Precision
    precision = round(precision_score(y_test, y_pred), 3)
    print("Precision  (Navie Baysain):", precision)
    Result.append("Precision (Navie Baysain)")
    Result.append("Precision (Navie Baysain):" + str(precision))
    # Precision
    recall = round(recall_score(y_test, y_pred), 3)
    print("Recall (Navie Baysain):", recall)
    Result.append("Recall (Navie Baysain)")
    Result.append("Recall (Navie Baysain):" + str(recall))
    # Confussion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    Result.append("Confusion Matrix (Navie Baysain)")
    Result.append("Confusion Matrix (Navie Baysain):" + str(c_matrix))
    # Classification report
    c_report = classification_report(y_test, y_pred)
    print(c_report)
    Result.append("Classification Report (Navie Baysain)")
    Result.append("Classification Report (Navie Baysain):" + str(c_report))
    plt.style.use('ggplot')

    x = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    performance_score = [None] * 4
    performance_score[0] = accuracy * 100
    performance_score[1] = f_score * 100
    performance_score[2] = precision * 100
    performance_score[3] = recall * 100

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance_score, color='#884EA0')
    plt.xlabel("Performance Measure")
    plt.ylabel("Score")
    plt.title("Performance Analysis of Naive Bayes")

    plt.xticks(x_pos, x)
    plt.savefig("NB_results")
    plt.show()
    return Result
