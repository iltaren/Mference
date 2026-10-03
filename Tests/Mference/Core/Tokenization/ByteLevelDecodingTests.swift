import Foundation
import Testing
@testable import Mference
import MferenceValidationSupport

/// The incremental byte-level path (Qwen's GPT-2-style BPE with clean-up off)
/// must stream exactly what the library's `decode(skipSpecialTokens: true)`
/// returns for the same IDs, while touching each token once.
@Suite("Byte-level lossless decode")
struct ByteLevelDecodingTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    @Test("A ByteLevel decoder with clean-up off takes the incremental path")
    func fixtureTakesTheIncrementalPath() {
        #expect(tok.byteLevelDecoding != nil)
    }

    @Test("Clean-up on keeps the library decode")
    func cleanUpKeepsTheLibraryPath() async throws {
        let library = try await Self.fixtureVariant(cleanUp: true)
        #expect(library.byteLevelDecoding == nil)
    }

    @Test("Random ID streams stream exactly the library decode")
    func randomStreamsMatchLibraryDecode() {
        Self.expectStreamsMatchLibrary(tok, seedLabel: "byte-level-specials", specials: [248044, 248046, 248068])
    }

    /// Real Qwen 3.6 ships `<think>`, `<tool_call>` and friends as added but not
    /// special tokens: the library keeps their text verbatim and decodes the
    /// byte run before them on its own, so a split scalar there becomes U+FFFD.
    @Test("Non-special added tokens are verbatim and end the byte run")
    func nonSpecialAddedTokensMatchLibraryDecode() async throws {
        let qwenLike = try await Self.fixtureVariant(cleanUp: false,
                                                     nonSpecial: ["<think>", "</think>", "<tool_call>"])
        #expect(qwenLike.byteLevelDecoding != nil)
        Self.expectStreamsMatchLibrary(qwenLike, seedLabel: "byte-level-added", specials: [248044, 248068, 248069, 248058])

        // An emoji split by <think>: the library renders the cut prefix as U+FFFD.
        let emoji: [Int32] = qwenLike.encode("🦙", addBOS: false)
        let letterA: Int32 = 97
        let think: Int32 = 248_068
        var ids: [Int32] = [letterA]
        ids.append(contentsOf: emoji.prefix(2))
        ids.append(think)
        ids.append(contentsOf: emoji.dropFirst(2))
        var detok = MFDetokenizer(tokenizer: qwenLike)
        var streamed = ""
        for id in ids { streamed += detok.push(id) }
        streamed += detok.flush()
        let library = qwenLike.tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: true)
        #expect(streamed == library)
        #expect(streamed.hasPrefix("a\u{FFFD}<think>"), "got \(streamed.debugDescription)")
    }

    /// Everything handed out before the tail stays handed out: a stream is
    /// append-only, so the concatenated deltas after every push are a prefix
    /// of the final text.
    @Test("Deltas never hold back a complete scalar")
    func completeScalarsAreEmittedImmediately() {
        let ids = tok.encode("a漢🦙b", addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var streamed = ""
        var afterEachPush: [String] = []
        for id in ids {
            streamed += detok.push(id)
            afterEachPush.append(streamed)
        }
        streamed += detok.flush()
        #expect(streamed == "a漢🦙b")
        #expect(afterEachPush.last == "a漢🦙b", "the last push completes the text without a flush")
    }

    // MARK: - Helpers

    private static func expectStreamsMatchLibrary(_ tok: MFTokenizer, seedLabel: String, specials: [Int32]) {
        var rng = SeedTree(0xB17E).key(seedLabel)
        for trial in 0..<400 {
            let length = 1 + Int(rng.next() % 40)
            var ids: [Int32] = []
            for _ in 0..<length {
                let roll = rng.next() % 100
                let id: Int32
                if roll < 70 {
                    // A raw byte: continuation, lead or ASCII.
                    id = Int32(rng.next() % 256)
                } else if roll < 80 {
                    id = Int32(256 + rng.next() % 2)
                } else if roll < 95 {
                    id = specials[Int(rng.next() % UInt64(specials.count))]
                } else {
                    // Outside the vocabulary: the library drops it.
                    id = Int32(300 + rng.next() % 50)
                }
                ids.append(id)
            }
            // Whole UTF-8 scalars, too, so valid multi-byte text is common.
            if trial % 4 == 0 {
                ids += tok.encode("é漢🦙\u{FFFD}", addBOS: false)
            }
            var detok = MFDetokenizer(tokenizer: tok)
            var streamed = ""
            for id in ids { streamed += detok.push(id) }
            streamed += detok.flush()
            let library = tok.tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: true)
            #expect(streamed == library, "trial \(trial) ids \(ids)")
        }
    }

    /// A temporary copy of the ChatML fixture with clean-up and the special
    /// flag of named added tokens changed.
    static func fixtureVariant(cleanUp: Bool, nonSpecial: Set<String> = []) async throws -> MFTokenizer {
        let source = try ChatMLTemplateTests.fixtureFolder()
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("byte-level-fixture-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for name in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"] {
            try FileManager.default.copyItem(at: source.appendingPathComponent(name),
                                            to: folder.appendingPathComponent(name))
        }
        let configURL = folder.appendingPathComponent("tokenizer_config.json")
        var config = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any])
        config["clean_up_tokenization_spaces"] = cleanUp
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)
        if !nonSpecial.isEmpty {
            let dataURL = folder.appendingPathComponent("tokenizer.json")
            var data = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: dataURL)) as? [String: Any])
            let added = try #require(data["added_tokens"] as? [[String: Any]])
            data["added_tokens"] = added.map { token -> [String: Any] in
                var token = token
                if let content = token["content"] as? String, nonSpecial.contains(content) {
                    token["special"] = false
                }
                return token
            }
            try JSONSerialization.data(withJSONObject: data).write(to: dataURL)
        }
        return try await MFTokenizer.load(from: folder)
    }
}
