import pandas as pd
from sklearn.model_selection import StratifiedKFold 

class CrossValidation:
    def __init__(
            self, 
            df, 
            target_cols, 
            problem_type="binary_classification", 
            num_folds =5,
            shuffle = True,
            random_state = 42,):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type == "binary_classification":
            unique_values = self.dataframe[self.target_cols[0]].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found for classification!")
            elif unique_values > 2:
                target = self.target_cols[0]
                kf = StratifiedKFold(n_splits=self.num_folds, 
                                    shuffle=self.shuffle, 
                                    random_state=self.random_state)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                    y= self.dataframe.target.values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        
        return self.dataframe


if __name__ == "__main__":
    cv = CrossValidation(df, target_cols=["target"])
