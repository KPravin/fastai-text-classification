import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(fastai_text_classificationTests.allTests),
    ]
}
#endif
