    import XCTest
    @testable import fastai_text_classification

    final class fastai_text_classificationTests: XCTestCase {
        func testExample() {
            // This is an example of a functional test case.
            // Use XCTAssert and related functions to verify your tests produce the correct
            // results.
            XCTAssertEqual(fastai_text_classification().text, "Hello, World!")
        }

        static var allTests = [
            ("testExample", testExample),
        ]
    }
