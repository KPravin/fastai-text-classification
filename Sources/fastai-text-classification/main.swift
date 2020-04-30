import SwiftAI
import PythonKit
import Foundation
import ArgumentParser

struct FastaiTextClassification: ParsableCommand {
    @Option(name: .shortAndLong, help: "Name of csv file with input data")
    var datasetKey: String

    @Option(name: .shortAndLong)
    var minLmEpochs: Int

    @Option(name: .shortAndLong)
    var maxLmEpochs: Int
    
    @Option(name: .shortAndLong)
    var minClassifierEpochs: Int

    @Option(name: .shortAndLong)
    var maxClassifierEpochs: Int

    @Option(name: .shortAndLong)
    var maxUnfrozenLayers: Int

    func run() throws {
        let dataWithoutHeaderPath = "Resources/\(datasetKey).csv"
        let dataWithHeaderPath = "Resources/\(datasetKey)-with-header.csv"
        let encoderKey = "\(datasetKey)-language-model"
        //let classifierKey = "\(datasetKey)-classifier"

        let fileManager = FileManager.default

        if !fileManager.fileExists(atPath: dataWithHeaderPath){
            let wrangler = DataWrangler(csv_path: dataWithoutHeaderPath)
            wrangler.describe()
            wrangler.save_csv(path: dataWithHeaderPath)
        } else {
            print("Omitting the description of the dataset...")
        }

        let wrapper = ModelWrapper(csv_path: dataWithHeaderPath)

        for n_epochs_language_model in 4...7{
            wrapper.init_language_model()
            let learning_rate = 0.1//wrapper.find_learning_rate_with_steepest_loss()!
            wrapper.fit(n_cycles: n_epochs_language_model, learning_rate: learning_rate)
            wrapper.save(path: encoderKey, just_encoder: true)

            for n_epochs_classifier in 4...6{
                for n_unfrozen_layers in 0...3{
                    wrapper.init_classifier()
                    let learning_rate = wrapper.find_learning_rate_with_steepest_loss()!
                    wrapper.load(path: encoderKey, just_encoder: true)
                    wrapper.freeze(n_layers: -n_unfrozen_layers)
                    wrapper.fit(n_cycles: n_epochs_classifier, learning_rate: learning_rate)
                    print("Interpretation of the results given by model having encoder trained during \(n_epochs_language_model) epochs, " +
                    "classifier during \(n_epochs_classifier) epochs and with \(n_unfrozen_layers) unfrozen layers:")
                    wrapper.interpret()
                }
            }
        }
    }
}

FastaiTextClassification.main()
