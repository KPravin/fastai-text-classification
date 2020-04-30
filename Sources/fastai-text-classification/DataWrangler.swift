// comment so that Colab does not interpret `#if ...` as a comment
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let pd = Python.import("pandas")

class DataWrangler {
    let csv_path: String
    let df: PythonObject

    init(csv_path: String) {
        self.csv_path = csv_path
        df = pd.read_csv(csv_path, header: Python.None).rename(
            columns: [0: "mark", 1: "id", 2: "time", 3: "device", 4: "user", 5: "text"]
        )
    }

    func drop_redundant_columns(columns_to_retain: String...) -> PythonObject{
        return df[columns_to_retain]
    }

    func describe(top_n: Int = 10) {
        print(#"Data is read from file "\#(csv_path)""#)
        print("Number of records: \(df.shape[0])")
        print("The first \(top_n) records from the collection:")
        print(df.head(top_n))
        print("The first \(top_n) records after removing redundant columns:")
        let concise_df = drop_redundant_columns(columns_to_retain: "mark", "text")
        print(concise_df.head(top_n))
        print("Some additional info about datates:")
        print(concise_df.info())
        print("Some statistics about marks:")
        print(concise_df.mark.describe())
        print("Distribution of tweet lengths:")
        print(concise_df.text.apply(Python.len).describe())
    }

    func save_csv(path: String) {
        df.to_csv(path)
    }
}
