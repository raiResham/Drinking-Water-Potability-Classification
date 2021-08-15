from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from os import path
import pandas as pd
import pickle


class Model:
    """
    A class to create,train and predict using RandomForestClassifier model.

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
        Trains and save RandomForestClassifier() model.


        Returns
        -------
        accuracy(int) : accuracy of the model.
        """

        self.preprocessing()
        self.split_train_test()

        model = RandomForestClassifier(n_estimators = 95, criterion = "gini", max_depth = 90)
        model.fit(self.x_train,self.y_train)
        self.pred = model.predict(self.x_test)
        accuracy =  accuracy_score(self.pred,self.y_test)

        # pickle trained model 
        with open("../saved_model/model","wb") as f:
            pickle.dump(model,f)
        return accuracy

    def predict(self, input):
        """
        Predicts the output

        Parameters:
        -----------
        input(list) : list of inputs

        Returns:
        --------
        pred(list) : list of predictions
        """

        # Check if the saved model exits
        if path.exists("../saved_model/model"):
            try:
                with open("../saved_model/model","rb") as f:
                    model = pickle.load(f)
                    pred = model.predict(input)
                    return pred
            except Exception as e:
                return "Exception occured :" + str(e)
        else:
            return "No model found! Please, train the model first."
    
