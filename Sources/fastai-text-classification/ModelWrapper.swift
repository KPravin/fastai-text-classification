// comment so that Colab does not interpret `#if ...` as a comment
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let ft = Python.import("fastai.text")

class ModelWrapper {
    let csv_path: String
    let learner_kind: PythonObject
    let batch_size: Int 
    let drop_mult: Double

    var language_model_databunch: PythonObject?
    var classifier_databunch: PythonObject?
    var learner: PythonObject?
    
    init(csv_path: String, batch_size: Int = 48, learner_kind: PythonObject = ft.AWD_LSTM, drop_mult: Double = 0.3) {
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.learner_kind = learner_kind
        self.drop_mult = drop_mult
    }

    func trace_learning_rates() {
        if let learner = learner {
            learner.lr_find()
            learner.recorder.plot()
        } else {
            print("Learner is not yet initialized")
        }
    }

    func freeze(n_layers: Int) {
        if let learner = learner {
            if n_layers == 0 {
                learner.unfreeze()
            } else {
                learner.freeze_to(n_layers)
            }
        } else {
            print("Learner is not yet initialized")
        }
    }

    func fit(n_cycles: Int = 5, learning_rate: Double = 0.1, momentums: [Double] = [0.8, 0.7]) {
        learner!.fit_one_cycle(n_cycles, learning_rate, moms: momentums)
    }

    func init_language_model() {
        language_model_databunch = ft.TextList.from_csv(".", csv_path, cols: "text").split_by_rand_pct(seed: 17).label_for_lm().databunch()
        learner = ft.language_model_learner(language_model_databunch, learner_kind, drop_mult: drop_mult)
    }

    func init_classifier() {
        classifier_databunch = ft.TextList.from_csv(".", csv_path, cols: "text", vocab: language_model_databunch!.vocab).split_by_rand_pct(seed: 17)
        .label_from_df(cols: "mark").databunch(bs: batch_size)
        learner = ft.text_classifier_learner(classifier_databunch!, learner_kind, drop_mult: drop_mult)
    }

    func save(path: String, just_encoder: Bool = false) {
        if let learner = learner {
            if just_encoder {
                learner.save_encoder(path)
            } else {
                learner.save(path)
            }
        } else {
            print("Learner is not yet initialized")
        }
    }

    func load(path: String, just_encoder: Bool = false) {
        if let learner = learner {
            if just_encoder {
                learner.load_encoder(path)
            } else {
                learner.load(path)
            }
        } else {
            print("Learner is not yet initialized")
        }
    }
}
