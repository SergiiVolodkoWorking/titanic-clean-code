from metaflow import step, FlowSpec
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from titanic_productization import transform


class TitanicMLFlow(FlowSpec):
    @step
    def start(self):
        print("Starting the pipeline")
        self.next(self.load)

    @step
    def load(self):
        self.train_df = pd.read_csv("data/train.csv")
        self.test_df = pd.read_csv("data/test.csv")
        self.next(self.transform_train)

    @step
    def transform_train(self):
        self.train_df = transform(self.train_df)
        self.train_df = self.train_df.drop(["Ticket", "Cabin", "Name", "Parch", "SibSp", "FamilySize", "PassengerId"], axis=1)
        self.next(self.transform_test)

    @step
    def transform_test(self):
        self.test_df = transform(self.test_df)
        self.test_df = self.test_df.drop(["Ticket", "Cabin", "Name", "Parch", "SibSp", "FamilySize", "PassengerId",], axis=1)
        self.next(self.configure_training)

    @step
    def configure_training(self):
        self.Y_train = self.train_df["Survived"]
        self.X_train = self.train_df.drop("Survived", axis=1)
        self.X_test = self.test_df.copy()
        self.next(self.train_svc, self.train_knn, self.train_random_forest)

    @step
    def train_svc(self):
        svc = SVC()
        svc.fit(self.X_train, self.Y_train)
        self.acc_svc = round(svc.score(self.X_train, self.Y_train) * 100, 2)
        self.next(self.wait_all_models)

    @step
    def train_knn(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.Y_train)
        self.acc_knn = round(knn.score(self.X_train, self.Y_train) * 100, 2)
        self.next(self.wait_all_models)

    @step
    def train_random_forest(self):
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.Y_train)
        random_forest.score(self.X_train, self.Y_train)
        self.acc_random_forest = round(random_forest.score(self.X_train, self.Y_train) * 100, 2)
        self.next(self.wait_all_models)

    @step
    def wait_all_models(self, inputs):
        self.merge_artifacts(inputs)
        models = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Random Forest'],
            'Score': [self.acc_svc, self.acc_knn, self.acc_random_forest]})
        print(models.sort_values(by='Score', ascending=False))
        self.next(self.end)

    @step
    def end(self):
        print('Pipeline is completed')

# Run the flow by executing:
# python ml_pipeline.py run
# to generate drawing of the pipeline
# For vertical graph layout:
# python ml_pipeline.py output-dot | dot -Grankdir=TB -Tpng -o graph.png
if __name__ == '__main__':
    TitanicMLFlow()