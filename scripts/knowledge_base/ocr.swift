// Local OCR for scanned PDFs. Input: JSON image manifest; output: one JSON page per line.
// swiftc -module-cache-path /tmp/alfa-swift-cache ocr.swift -o /tmp/alfa-ocr
import Foundation
import Vision
import ImageIO

struct InputPage: Decodable {
    let source: String
    let page: Int
    let image: String
    let width: Int
    let height: Int
}
struct Line: Encodable {
    let text: String
    let confidence: Float
    // Normalized coordinates, origin at the TOP LEFT.
    let bbox: [Double]
}
struct OutputPage: Encodable {
    let source: String
    let page: Int
    let width: Int
    let height: Int
    let engine: String
    let revision: Int
    let lines: [Line]
}

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: alfa-ocr images.json pages.jsonl\n", stderr)
    exit(2)
}
let inputs = try JSONDecoder().decode([InputPage].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard !FileManager.default.fileExists(atPath: output.path) else {
    fputs("Refusing to overwrite OCR output; choose a new path.\n", stderr)
    exit(2)
}
FileManager.default.createFile(atPath: output.path, contents: nil)
let handle = try FileHandle(forWritingTo: output)
defer { try? handle.close() }
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
for input in inputs {
    try autoreleasepool {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["ru-RU", "en-US"]
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.002
        let handler = VNImageRequestHandler(url: URL(fileURLWithPath: input.image), options: [:])
        try handler.perform([request])
        let lines = (request.results ?? []).compactMap { observation -> Line? in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            let b = observation.boundingBox
            return Line(text: candidate.string, confidence: candidate.confidence,
                        bbox: [b.minX, 1 - b.maxY, b.maxX, 1 - b.minY])
        }
        guard !lines.isEmpty else {
            throw NSError(domain: "OCR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Empty page: \(input.source) p\(input.page)"])
        }
        let page = OutputPage(source: input.source, page: input.page, width: input.width,
                              height: input.height, engine: "Apple Vision accurate ru-RU/en-US",
                              revision: request.revision, lines: lines)
        try handle.write(contentsOf: encoder.encode(page))
        try handle.write(contentsOf: Data([10]))
        fputs("\(input.source) p\(input.page): \(lines.count) lines\n", stderr)
    }
}
