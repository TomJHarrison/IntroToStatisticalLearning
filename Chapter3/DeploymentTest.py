from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

# --------- Model Training ---------
# Load Iris dataset
iris = load_iris()

# Extract features and labels
X, y = iris.data, iris.target

# Train the model
clf = GaussianNB()
clf.fit(X, y)


# --------- FastAPI Deployment ---------
# Create FastAPI instance
app = FastAPI()

# Define Pydantic model for input features
class IrisFeatures(BaseModel):
    """
    Pydantic model for input features of the Iris dataset.

    Later we will use this model to validate the input data for our prediction endpoint.
    This ensures that the data we receive is in the correct format, contains all the necessary features,
    and ensures the variables are of the correct type.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: IrisFeatures):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return {"class": iris.target_names[class_idx]}

@app.get("/")
def health_check():
    return {"status": "ok"}