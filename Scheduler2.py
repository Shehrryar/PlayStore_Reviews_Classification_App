from __future__ import print_function


import tkinter as form
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter import messagebox as msg
from tkinter import *
from time import gmtime, strftime
import json, csv
import sys
import copy
from random import randint
import os
import time

import top as top

import UserInput
import UserInputForm
from globalData import *
from algorithms import *
from UserInputForm import *
import seaborn as sns
import pandas as pd
from tkinterhtml import HtmlFrame
from urllib.request import urlopen
from pandastable import Table
from tkintertable import TableCanvas
from PIL import ImageTk,Image
class Scheduler2(form.Frame):

    def __init__(self, master):
        form.Frame.__init__(self, master)

        master.title("PlayStore Reviews Detection")
        #center window in the middle of the screen
        master.geometry("%dx%d+%d+%d" % (590,650, int(master.winfo_screenwidth() / 2 - 700 / 2), int(master.winfo_screenheight() / 2 - 600 / 2)))
        master.minsize(width=1000, height=250)
        master.resizable(width=False, height=False)
        #master.mainloop()
        menuBar = form.Menu(master)
        master.config(menu=menuBar)

        #self.dataFrame = Frame(self.master, width=1000, height=400)
        #self.dataFrame.pack()
        #self.dataFrame.grid(columnspan=8, row=7, padx=5, pady=5)

        tabController = ttk.Notebook(master, width=40)
        frTabMain = ttk.Frame(tabController)
        
        self.frTabRandomSol = ttk.Frame(tabController)
        self.frTabFEF = ttk.Frame(tabController)
        self.frTabPrediction = ttk.Frame(tabController)
        self.frTabHybrid = ttk.Frame(tabController)
        self.frTabOptimal = ttk.Frame(tabController)
        self.frTabGanntChart1 = ttk.Frame(tabController)
        self.frTabGanntChart2 = ttk.Frame(tabController)

        self.frTabHistogram = ttk.Frame(tabController)
        self.frTabScatter = ttk.Frame(tabController)
        
        tabController.add(frTabMain,  text="Main")
        tabController.add(self.frTabFEF,  text="Logistic Regression (LR) Results")
        tabController.add(self.frTabPrediction,  text="Support Vector Machine (SVM) Results")
        tabController.add(self.frTabHybrid, text="Random Forest (RF) Results")
        tabController.add(self.frTabOptimal, text="Naive Baysain (NB) Results")
        # tabController.add(self.frTabGanntChart1, text="Hybrid Machine Visualization")
        # tabController.add(self.frTabGanntChart2, text="Optimized Hybrid Machine Visualization")


        tabController.pack(expand=1, fill="both")

        ttk.Button(frTabMain, text="Load Data", width=32, command=self.loadData).grid(column=0, row=0, padx=5, pady=5)
        ttk.Button(frTabMain, text="Data Analysis", width=32,command=self.dataAnalysis).grid(column=0, row=1, padx=5, pady=5)
        ttk.Button(frTabMain, text="Predict your review", width=32, command=self.predict_class).grid(column=0, row=2, padx=5,
                                                                                              pady=5)
        #self.message_label.grid(row=7, column=4)
        #self.f=ttk.Frame(frTabMain, height=1500, width=300)
        #self.f.grid(row=4, column=1)
        #self.f = Frame(self.master, height=200, width=300)
        #self.f.grid()

        
        form.Button(frTabMain, text="Logistic Regression", width=32, height=4, bg="#808000", font = ('Arial', 8,'bold'), fg='#ffffff',  command=self.Run1).grid(column=0, row=3, padx=5, pady=5)
        self.calculationProgressBar1 =ttk.Progressbar(frTabMain, orient="horizontal", length = 230, mode="determinate")
        self.calculationProgressBar1.grid(column=0, row=4, padx=5, pady=5)
        self.calculationProgressBar1["maximum"] = 100
        
        self.lblCalculationsRes1  = ttk.Label(frTabMain)
        self.lblCalculationsRes1.grid(column=0, row=5, padx=5, pady=5)
        
        form.Button(frTabMain, text="Support Vector Machine", width=32, height=4, bg="#008080", font = ('Arial', 8,'bold'), fg='#ffffff', command=self.Run2).grid(column=1, row=3, padx=5, pady=5)
        self.calculationProgressBar2 =ttk.Progressbar(frTabMain, orient="horizontal", length = 230, mode="determinate")
        self.calculationProgressBar2.grid(column=1, row=4, padx=5, pady=5)
        self.calculationProgressBar2["maximum"] = 100

        self.lblCalculationsRes2  = ttk.Label(frTabMain)
        self.lblCalculationsRes2.grid(column=1, row=5, padx=5, pady=5)
        
        form.Button(frTabMain, text="Random Forest", width=32, height=4, bg="#F08080", font = ('Arial', 8,'bold'), fg='#ffffff', command=self.Run3).grid(column=2, row=3, padx=5, pady=5)
        self.calculationProgressBar3 =ttk.Progressbar(frTabMain, orient="horizontal", length = 230, mode="determinate")
        self.calculationProgressBar3.grid(column=2, row=4, padx=5, pady=5)
        self.calculationProgressBar3["maximum"] = 100

        self.lblCalculationsRes3  = ttk.Label(frTabMain)
        self.lblCalculationsRes3.grid(column=2, row=5, padx=5, pady=5)
        
        form.Button(frTabMain, text="Navie Baysain", width=32, height=4, bg="#5499C7", font = ('Arial', 8,'bold'), fg='#ffffff', command=self.Run4).grid(column=3, row=3, padx=5, pady=5)
        self.calculationProgressBar4 =ttk.Progressbar(frTabMain, orient="horizontal", length = 230, mode="determinate")
        self.calculationProgressBar4.grid(column=3, row=4, padx=5, pady=5)
        self.calculationProgressBar4["maximum"] = 100

        self.lblCalculationsRes4  = ttk.Label(frTabMain)
        self.lblCalculationsRes4.grid(column=3, row=5, padx=5, pady=5)

        self.lblApplicationCount = ttk.Label(frTabMain)
        self.lblApplicationCount.grid(column=1, row=0, padx=5, pady=5)
        self.lblReviewsCount= ttk.Label(frTabMain)
        self.lblReviewsCount.grid(column=1, row=1, padx=5, pady=5)

        self.lblSpamCount = ttk.Label(frTabMain)
        self.lblSpamCount.grid(column=2, row=0, padx=5, pady=5)
        self.lblNotSpamReviewsCount = ttk.Label(frTabMain)
        self.lblNotSpamReviewsCount.grid(column=2, row=1, padx=5, pady=5)

        self.lblPerSpamCount = ttk.Label(frTabMain)
        self.lblPerSpamCount.grid(column=3, row=0, padx=5, pady=5)
        self.lblPerNotSpamReviewsCount = ttk.Label(frTabMain)
        self.lblPerNotSpamReviewsCount.grid(column=3, row=1, padx=5, pady=5)

        self.message_label = ttk.Label(frTabMain, font = ('Arial', 14,'underline'))
        self.message_label.grid(columnspan=6, row=6, padx=5, pady=5)


        # self.dataFrame= ttk.Frame(frTabMain, width = 1000, height = 400, border=5)
        # self.dataFrame.grid(columnspan=8,row=7, padx=5, pady=5)


        # self.message_label2 = ttk.Label(self.master,
        #                        text='Input Features',
        #                        font=('Arial', 14, 'underline'),
        #                        fg='Red').grid(column=3, row=7)
        # self.message_label.pack()
        # self.message_label2.pack()



        

        ###########

        frmFEF = ttk.LabelFrame(self.frTabFEF, text="Classification Output (Logistic Regression Model)")
        frmFEF.grid(column = 0, row =0, padx=8, pady=8)
        
        scrollbar = ttk.Scrollbar(frmFEF)
        self.lboxFEF = form.Listbox(frmFEF, width=90, height=20, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar.set, selectmode=form.SINGLE)
        self.lboxFEF.grid(columnspan=3, column=0, row=0, padx=3, pady=3)

        ###########

        frmLeanedFEF = ttk.LabelFrame(self.frTabPrediction, text="Classification Output (Support Vector Machine Model)")
        frmLeanedFEF.grid(column = 0, row =0, padx=8, pady=8)
        
        scrollbar = ttk.Scrollbar(frmLeanedFEF)
        self.lboxLearnFEF = form.Listbox(frmLeanedFEF, width=90, height=20, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar.set, selectmode=form.SINGLE)
        self.lboxLearnFEF.grid(columnspan=3, column=0, row=0, padx=3, pady=3)

        ###########

        frmHybrid = ttk.LabelFrame(self.frTabHybrid, text="Classification Output (Random Forest Model)")
        frmHybrid.grid(column = 0, row =0, padx=8, pady=8)
        
        scrollbar = ttk.Scrollbar(frmHybrid)
        self.lboxHybrid = form.Listbox(frmHybrid, width=90, height=20, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar.set, selectmode=form.SINGLE)
        self.lboxHybrid.grid(columnspan=3, column=0, row=0, padx=3, pady=3)

        ###########

        frmLeanedHybrid = ttk.LabelFrame(self.frTabOptimal, text="Classification Output (Navie Baysain Model)")
        frmLeanedHybrid.grid(column = 0, row =0, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(frmLeanedHybrid)
        self.lboxLeanedHybrid = form.Listbox(frmLeanedHybrid, width=90, height=20, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar.set, selectmode=form.SINGLE)
        self.lboxLeanedHybrid.grid(columnspan=3, column=0, row=0, padx=3, pady=3)


        global load_reviews_data

    def dataAnalysis(self):
        data_path = 'dataset/final_input_data.csv'
        load_reviews_data = pd.read_csv(data_path)
        load_reviews_data.drop(['review_id'], axis=1, inplace=True)
        load_reviews_data.drop(['app_id'], axis=1, inplace=True)
        #load_reviews_data.drop(['class_label'], axis=1, inplace=True)

        #Correlation Analysis
        plt.figure(figsize=(15, 15))
        sns.heatmap(load_reviews_data.corr(), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
        plt.show()
        # Review Rating Score Analysis
        sns.countplot(x='review_score', data=load_reviews_data, palette='hls')
        plt.savefig('review_score')
        plt.show()
        # Class Label Analysis
        sns.countplot(x='class_label', data=load_reviews_data, palette='hls')
        plt.savefig('data_distribution')
        plt.show()

        # Comparision of Positive and Negative Words
        plt.figure(figsize=(20, 8))
        x_dot = [x for x in range(500)]
        plt.plot(x_dot, load_reviews_data.total_positive_words[:500], c="blue", linewidth=3, label="Positive Words")
        plt.plot(x_dot, load_reviews_data.total_negative_words[:500], c="red", linestyle="dashed", linewidth=3,
                 label="Negative Words")
        plt.legend()
        plt.title("Positive Words  Vs Negative Words ", fontsize=20)
        plt.ylabel('Positive Vs Negative Words', size=18)
        plt.xlabel('Iterations', size=18)
        plt.show()
        # Comparision of Positive Words Polarity and Negative Words Polarity
        plt.figure(figsize=(20, 8))
        x_dot = [x for x in range(500)]
        plt.plot(x_dot, load_reviews_data.total_positive_polarity[:500], c="blue", linewidth=3, label="Positive Words Polarity")
        plt.plot(x_dot, load_reviews_data.total_negative_polarity[:500], c="red", linestyle="dashed", linewidth=3,
                 label="Negative Words Polarity")
        plt.legend()
        plt.title("Positive Words Polarity  Vs Negative Words Polarity ", fontsize=20)
        plt.ylabel('Positive Vs Negative Words Polarity', size=18)
        plt.xlabel('Iterations', size=18)
        plt.show()
        # Comparision of Positive Ratio and Negative Ratio
        plt.figure(figsize=(20, 8))
        x_dot = [x for x in range(500)]
        plt.plot(x_dot, load_reviews_data.p_ratio[:500], c="blue", linewidth=3, label="Positive Ratio")
        plt.plot(x_dot, load_reviews_data.n_ratio[:500], c="red", linestyle="dashed", linewidth=3, label="Negative Ratio")
        plt.legend()
        plt.title("Positive Ratio  Vs Negative Ratio ", fontsize=20)
        plt.ylabel('Positive Vs Negative', size=18)
        plt.xlabel('Iterations', size=18)
        plt.show()




    def display_xls_file(self):
        data_path = 'dataset/final_input_data.csv'
        load_reviews_data = pd.read_csv(data_path)
        print("Hello")
        try:
            if (len(load_reviews_data) == 0):
                msg.showinfo('No records', 'No records')
            else:
                pass

            # Now display the DF in 'Table' object
            # under'pandastable' module

            f2= Frame(self.master,height=1000, width=300)
            #f2 = Frame(self.master, height=200, width=300)
            #f2.grid(row=6)
            f2.pack(fill=BOTH, expand=1)
            table = Table(f2, dataframe=load_reviews_data, read_only=True)
            table.show()
            self.message_label.config(text=STRGS['REVIEWS_DATA'])

        except FileNotFoundError as e:
            print(e)
            msg.showerror('Error in opening file', e)
    def loadData(self):   ## DONE
        global load_reviews_data
        data_path = 'dataset/final_input_data.csv'
        load_reviews_data = pd.read_csv(data_path)
        print("Total Number of Reviews:", len(load_reviews_data) )
        print(load_reviews_data.head())

        unique_applications= load_reviews_data.app_id.unique()
        print("Total Number of Unique Applications:", len(unique_applications))

        print(load_reviews_data['class_label'].value_counts())

        count_no_sub = len(load_reviews_data[load_reviews_data['class_label'] == 0])
        count_sub = len(load_reviews_data[load_reviews_data['class_label'] == 1])

        per_spam_reviews= round((count_sub/len(load_reviews_data))*100,3)
        per_nospam_reviews = round((count_no_sub / len(load_reviews_data)) * 100,3)


        #self.dataAnalysis()


        self.display_xls_file()
        msg.showinfo("Reviews Dataset", "Reviews data Successfully Loaded")
        pass

        self.lblApplicationCount.config(text= str(len(unique_applications)) + " Unique Applications")
        self.lblReviewsCount.config(text= str(len(load_reviews_data)) + " Unique Reviews" )
        self.lblSpamCount.config(text= str(count_sub) + " Totol Number of Spam Reviews ")
        self.lblNotSpamReviewsCount.config(text=str(count_no_sub) + " Totol Number of Real Reviews")

        self.lblPerSpamCount.config(text= str(per_spam_reviews) + "%  Spam Reviews ")
        self.lblPerNotSpamReviewsCount.config(text=str(per_nospam_reviews) + "% Real Reviews")



    def exitProgram(self):
        """Quit everything"""
        answer = msg.askyesno(STRGS['EXIT'], STRGS['MSG_REALLY_QUIT'])
        if answer:
            self.quit()
            self.destroy()
            exit()

    def Run1(self):
        global load_reviews_data
        predicted = 0
        print("Data Loaded:", len(load_reviews_data))
        if len(load_reviews_data):

            strTime = "Calculations time:\n"
            #jobList = prepareJobs()
            self.calculationProgressBar1["value"] = 100* 3/6
            self.calculationProgressBar1.update()
            startTime = time.perf_counter()

            Result = logisticRegression(copy.deepcopy(load_reviews_data), predicted)
            for res in range(0,len(Result)):
                self.lboxFEF.insert(form.END, Result[res])
                
            self.calculationProgressBar1["value"] = 100* 5/6
            self.calculationProgressBar1.update() 
            #startTime = time.perf_counter()
            #optResult = optimalSolution(copy.deepcopy(jobList),learned)
            strTime = strTime+"Logistic Regression (LR): "+ str(round(time.perf_counter() - startTime, 4))+"\n"
            #createGanttChart(self.frTabOptimal, optResult)

            self.lblCalculationsRes1.configure(text=strTime)

            self.calculationProgressBar1["value"] = 100* 6/6
            self.calculationProgressBar1.update() 

            msg.showinfo(STRGS['OK'], "Calculations finished!")
           
        else:
            msg.showerror(STRGS['ERR_ILLEGAL'], STRGS['MSG_ERR_EMPTY_VAL'])

    def Run2(self):
        global load_reviews_data
        predicted = 1
        if len(load_reviews_data):

            strTime = "Calculations time:\n"
            self.calculationProgressBar2["value"] = 100* 3/6
            self.calculationProgressBar2.update()
            startTime = time.perf_counter()

            Result = supportVectorMachine(copy.deepcopy(load_reviews_data), predicted)
            for res in range(0,len(Result)):
                self.lboxLearnFEF.insert(form.END, Result[res])

            self.calculationProgressBar2["value"] = 100* 5/6
            self.calculationProgressBar2.update() 
            #startTime = time.perf_counter()
            #optResult = optimalSolution(copy.deepcopy(jobList),predicted)
            strTime = strTime+"Support Vector Machine (SVM): "+ str(round(time.perf_counter() - startTime, 4))+"\n"
            #createGanttChart(self.frTabOptimal, optResult)

            self.lblCalculationsRes2.configure(text=strTime)

            self.calculationProgressBar2["value"] = 100* 6/6
            self.calculationProgressBar2.update() 

            msg.showinfo(STRGS['OK'], "Calculations finished!")
           
        else:
            msg.showerror(STRGS['ERR_ILLEGAL'], STRGS['MSG_ERR_EMPTY_VAL'])

    def Run3(self):
        global load_reviews_data
        predicted = 3
        if len(load_reviews_data):

            strTime = "Calculations time:\n"
            self.calculationProgressBar3["value"] = 100* 3/6
            self.calculationProgressBar3.update()
            startTime = time.perf_counter()

            self.calculationProgressBar3["value"] = 100* 5/6
            self.calculationProgressBar3.update() 
            startTime = time.perf_counter()

            result = randomForest(copy.deepcopy(load_reviews_data), predicted)

                    
            for res in range(0,len(result)):
                self.lboxHybrid.insert(form.END, result[res])

                          
            #strTime = strTime+"Random Forest: "+ str(round(time.perf_counter() - startTime, 4))+"\nMax CPU Unit Response Time=="+ str(max([j.endTime for j in optResult]))+"\n"

            strTime = strTime + "Random Forest (RF): " + str(round(time.perf_counter() - startTime, 4)) + "\n"
            #createGanttChart(self.frTabGanntChart1, optResult)

            self.lblCalculationsRes3.configure(text=strTime)

            self.calculationProgressBar3["value"] = 100* 6/6
            self.calculationProgressBar3.update() 

            msg.showinfo(STRGS['OK'], "Calculations finished!")
           
        else:
            msg.showerror(STRGS['ERR_ILLEGAL'], STRGS['MSG_ERR_EMPTY_VAL'])

    def predict_class(self):
        UserInputForm.main().wait_window()

    def Run4(self):
        global load_reviews_data
        learned = 4
        if len(load_reviews_data):

            strTime = "Calculations time:\n"
            self.calculationProgressBar4["value"] = 100* 3/6
            self.calculationProgressBar4.update()
            startTime = time.perf_counter()

            self.calculationProgressBar4["value"] = 100* 5/6
            self.calculationProgressBar4.update() 
            startTime = time.perf_counter()
            result = navieBaysain(copy.deepcopy(load_reviews_data), learned)

            for res in range(0,len(result)):
                self.lboxLeanedHybrid.insert(form.END, result[res])

            strTime = strTime + "Navie Baysain (NB): " + str(round(time.perf_counter() - startTime, 4)) + "\n"


            self.lblCalculationsRes4.configure(text=strTime)

            self.calculationProgressBar4["value"] = 100* 6/6
            self.calculationProgressBar4.update() 

            msg.showinfo(STRGS['OK'], "Calculations finished!")
           
        else:
            msg.showerror(STRGS['ERR_ILLEGAL'], STRGS['MSG_ERR_EMPTY_VAL'])
            
