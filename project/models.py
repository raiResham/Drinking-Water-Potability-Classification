from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow 

class Models:
   """
   A class to create and train different models.


   """

    def __init__(self):  
        """
        Loads csv file into dataframe.
        """

        self.water_df = pd.read_csv("../data/water_potability.csv")
        
    def preprocessing(self):
        """
        Drops rows from dataframe if it has NaN value.

        Returns
        -------
        None
        """

        self.water_df.dropna(how="any", inplace = True)

    def split_train_test(self):
        """
        Splits dataframe into train and test set.

        Returns
        -------
        None
        """

        self.X = self.water_df.drop(['Potability'], axis = 1)
        self.y = self.water_df["Potability"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.25,random_state=77)
        
    def train(self):
        """
        Trains multiple models like LogisticRegression(),KNeighborsClassifier() and RandomForestClassifier().
        Keep Log of the parameters and accuracy of each models with mlflow.

        Returns
        -------
        None
        """

        self.preprocessing()
        self.split_train_test()

        with mlflow.start_run():
            # Logistic Regression
            model = LogisticRegression()
            model.fit(self.x_train,self.y_train)
            self.pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.pred,self.y_test)
            mlflow.log_metric("accuracy",accuracy)
            mlflow.sklearn.log_model(model, "model")


        # KNearestNeighbors
        n_neighbors = range(1,15)
        for neighbors in range(len(n_neighbors)):
            neighbors = n_neighbors[neighbors]
            model = KNeighborsClassifier(n_neighbors = neighbors)
            model.fit(self.x_train,self.y_train)
            self.pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.pred,self.y_test)

            with mlflow.start_run():
                mlflow.log_param("n_neighbors", neighbors)
                mlflow.log_metric("accuracy",accuracy)
                mlflow.sklearn.log_model(model, "model")
            
        
        # RandomForestClassifier()
        n_estimators = range(95,110)
        criterions = ["gini", "entropy"]
        max_depth_list = [50, 60, 70, 80, 90, 100, None]

        for estimator in n_estimators:
            for criterion in criterions:
                for max_depth in max_depth_list:
                    model = RandomForestClassifier(n_estimators = estimator, criterion = criterion, max_depth = max_depth)
                    model.fit(self.x_train,self.y_train)
                    self.pred = model.predict(self.x_test)
                    accuracy =  accuracy_score(self.pred,self.y_test)
                    with mlflow.start_run():

                        mlflow.log_param("n_estimators", estimator)
                        mlflow.log_param("criterion", criterion)
                        mlflow.log_param("max_depth", max_depth)
                        mlflow.log_metric("accuracy",accuracy)
                        mlflow.sklearn.log_model(model, "model")
