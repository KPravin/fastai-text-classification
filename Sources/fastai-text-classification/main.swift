import SwiftAI
import PythonKit
import Foundation

struct fastai_text_classification {
    var text = "Hello, World!"
}

// print("ok")

// let datasetKey = "data"

// let dataWithoutHeaderPath = "Resources/\(datasetKey).csv"
// let dataWithHeaderPath = "Resources/\(datasetKey)-with-header.csv"
// let encoderKey = "\(datasetKey)-language-model"
// let classifierKey = "\(datasetKey)-classifier"

// let fileManager = FileManager.default

// if !fileManager.fileExists(atPath: dataWithHeaderPath){
//     let wrangler = DataWrangler(csv_path: dataWithoutHeaderPath)
//     wrangler.describe()
//     wrangler.save_csv(path: dataWithHeaderPath)
// } else {
//     print("Omitting the description of the dataset...")
// }

// let wrapper = ModelWrapper(csv_path: dataWithHeaderPath)
// wrapper.init_language_model()
// wrapper.fit()
// wrapper.save(path: encoderKey, just_encoder: true)
// wrapper.init_classifier()
// wrapper.load(path: encoderKey, just_encoder: true)
// wrapper.fit()
// wrapper.save(path: classifierKey)

// if !fileManager.fileExists(atPath: encoderPath){
//     wrapper.init_language_model()
//     wrapper.fit()
//     wrapper.save(path: encoderPath, just_encoder: true)
//     //wrapper.save_language_model(path: encoderPath)
// } else {
//     print("Omitting the language model training...")
// }

//print(wrapper.language_model!.get_batch())