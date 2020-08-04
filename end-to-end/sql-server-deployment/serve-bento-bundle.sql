sp_configure

EXEC sp_configure  'external scripts enabled', 1
RECONFIGURE WITH OVERRIDE

CREATE DATABASE irissql
GO

USE irissql
GO

DROP TABLE IF EXISTS iris_data;
GO


CREATE TABLE iris_data (
  id INT NOT NULL IDENTITY PRIMARY KEY
  , "Sepal.Length" FLOAT NOT NULL, "Sepal.Width" FLOAT NOT NULL
  , "Petal.Length" FLOAT NOT NULL, "Petal.Width" FLOAT NOT NULL
  , "Species" VARCHAR(100) NOT NULL, "SpeciesId" INT NOT NULL
);




CREATE PROCEDURE get_iris_dataset
AS
BEGIN
EXEC sp_execute_external_script @language = N'Python', 
@script = N'
from sklearn import datasets
iris = datasets.load_iris()
iris_data = pandas.DataFrame(iris.data)
iris_data["Species"] = pandas.Categorical.from_codes(iris.target, iris.target_names)
iris_data["SpeciesId"] = iris.target
', 
@input_data_1 = N'', 
@output_data_1_name = N'iris_data'
WITH RESULT SETS (("Sepal.Length" float not null, "Sepal.Width" float not null, "Petal.Length" float not null, "Petal.Width" float not null, "Species" varchar(100) not null, "SpeciesId" int not null));
END;
GO


INSERT INTO iris_data ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species", "SpeciesId")
EXEC dbo.get_iris_dataset;




CREATE PROCEDURE predict_species (@model VARCHAR(100))
AS
BEGIN
    DECLARE @svm_model VARBINARY(max)

    EXECUTE sp_execute_external_script @language = N'Python'
        , @script = N'
import bentoml
saved_path=r"C:\Program Files\Microsoft SQL Server\MSSQL15.NEWSERVER\bento_bundle"
irismodel = bentoml.load(saved_path)
species_pred = irismodel.predict(iris_data[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]])
iris_data["PredictedSpecies"] = species_pred
OutputDataSet = iris_data[["id", "SpeciesId", "PredictedSpecies"]] 
print(OutputDataSet)
'
        , @input_data_1 = N'select id, "Sepal.Length", "Sepal.Width",
            "Petal.Length", "Petal.Width",
            "SpeciesId" from iris_data'
        , @input_data_1_name = N'iris_data'
        , @params = N'@svm_model varbinary(max)'
        , @nb_model = @svm_model
WITH RESULT SETS((
              "id" INT
            , "SpeciesId" INT
            , "SpeciesId.Predicted" INT
            ));
END;
GO

EXECUTE predict_species 'SVM';
GO
