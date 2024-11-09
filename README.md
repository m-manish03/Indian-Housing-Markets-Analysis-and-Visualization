# Indian Housing Markets Analysis and Visualization

#### Team Members:
    M. Sri Manish Reddy (21911A05G7)
    B. Nilesh (21911A05D7)
    B. Rakesh Kumar (21911A05D9)
    B. Srikanth (21911A05E2)
#### Project Guide:
    Ms. A. Lalitha
#### Project Coordinators:
    Ms. G. Surekha
    Ms. K. Spandana Kumari

This project deals with Exploratory Data Analysis and Visualization of housing markets in major Indian Cities of Bengaluru, Chennai, Delhi, Hyderabad and Mumbai using regression models of:
1. Linear Regression
2. Polynomial Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Support Vector Regression
6. Extreme Gradient Boosting

The dataset is taken from Kaggle - https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india/code?datasetId=846956&sortBy=voteCount

@startuml
!include <archimate/Archimate>

Grouping(HousingAnalysisProject, "Indian Housing Markets Analysis and Visualization") {
    Technology_Node(dataCollectionNode, "Data Collection Node")
    Technology_Node(dataPreprocessingNode, "Data Preprocessing Node")
    Technology_Node(edaNode, "Exploratory Data Analysis Node")
    Technology_Node(mlModelsNode, "Machine Learning Models Node")
    Technology_Node(visualizationNode, "Visualization & Reporting Node")

    Technology_Artifact(csvFiles, "Dataset")
    Technology_Function(dataCleaning, "Data Cleaning")
    Technology_Function(dataEncoding, "Data Encoding")
    Technology_Function(dataScaling, "Data Scaling")
    Technology_Function(statisticalAnalysis, "Statistical Analysis")
    Technology_Function(modelTraining, "Model Training")
    Technology_Function(reportGeneration, "Report Generation")

    dataCollectionNode -down-> csvFiles : "collects"
    dataCollectionNode -down-> dataPreprocessingNode : "passes data to"

    dataPreprocessingNode -down-> dataCleaning : "includes"
    dataPreprocessingNode -down-> dataEncoding : "includes"
    dataPreprocessingNode -down-> dataScaling : "includes"
    dataPreprocessingNode -down-> edaNode : "outputs to"

    edaNode -down-> statisticalAnalysis : "includes"
    edaNode -down-> mlModelsNode : "feeds into"

    mlModelsNode -down-> modelTraining : "trains"
    mlModelsNode -down-> visualizationNode : "provides results to"
    
    visualizationNode -down-> reportGeneration : "presents"
}

@enduml
