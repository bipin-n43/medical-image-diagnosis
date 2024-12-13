import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel

# Initialize Spark Session inside Streamlit
spark = SparkSession.builder.appName("ImageClassificationApp").getOrCreate()

# Load the saved models
lr_model = LogisticRegressionModel.load("best_lr_model")
rf_model = RandomForestClassificationModel.load("best_rf_model")

st.title("Image Feature Classification")

# User Inputs
image_size_norm = st.number_input("Image Size (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
mean_intensity_norm = st.number_input("Mean Intensity (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
std_intensity_norm = st.number_input("Std Intensity (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest"])

if st.button("Predict"):
    # Create a PySpark DataFrame from user input
    schema = StructType([
        StructField("image_size_normalized", FloatType(), True),
        StructField("mean_intensity_normalized", FloatType(), True),
        StructField("std_intensity_normalized", FloatType(), True)
    ])

    input_data = [(float(image_size_norm), float(mean_intensity_norm), float(std_intensity_norm))]
    input_df = spark.createDataFrame(input_data, schema=schema)

    # VectorAssembler step must be applied as original pipeline step
    # If original models were part of a pipeline including assembler, reload the pipeline.
    # For demonstration, we reconstruct the assembler:
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(
        inputCols=["image_size_normalized", "mean_intensity_normalized", "std_intensity_normalized"],
        outputCol="features"
    )
    input_prepared = assembler.transform(input_df)

    # Predict using chosen model
    if model_choice == "Logistic Regression":
        predictions = lr_model.transform(input_prepared)
    else:
        predictions = rf_model.transform(input_prepared)

    prediction_row = predictions.collect()[0]
    st.write(f"Predicted Label: {prediction_row['prediction']}")