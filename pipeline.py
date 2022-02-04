from metaflow import step, FlowSpec
import pandas as pd

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
        self.test_df = self.test_df.drop(["Ticket", "Cabin", "Name", "Parch", "SibSp", "FamilySize"], axis=1)
        self.next(self.end)

    @step
    def end(self):
        print('Pipeline is completed')

# Run the pipline by executing:
# python pipeline.py run
# to generate drawing of the pipeline
# For vertical graph layout:
# python pipeline.py output-dot | dot -Grankdir=TB -Tpng -o graph.png
if __name__ == '__main__':
    TitanicMLFlow()