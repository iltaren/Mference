import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference

/// Quality gate for Qwen 3.6 prefill kernels that reorder floating-point sums.
///
/// Set `MFERENCE_QWEN_PREFILL_GATE` to an installed `qwen36.gturbo`; the
/// install is read-only. Three prefills of the same tokens: control (tiled
/// full attention, per-row routed experts), attention (tensor-ops full
/// attention, per-row experts) and candidate (the default: tensor-ops
/// attention, grouped-GEMM experts). Each is teacher-forced through the same
/// tokens, so every difference comes from the prefill kernels; control ->
/// attention isolates the attention kernel and attention -> candidate the
/// grouped experts.
///
/// Items: the frozen community prompts with the saved Gemma answers (the
/// answers are inputs, not expectations), plus, when
/// `MFERENCE_QWEN_PREFILL_GATE_LONG` names a chat messages file, its prompt
/// with the last 200 prompt tokens scored instead of prefilled. Pass rule as in
/// `GemmaPrefillEquivalenceGateTests`: fail on a mean shift > 0.005 nats per
/// token that is distinguishable from zero, any shift > 0.02, or top-1
/// agreement below 98 %. `MFERENCE_QWEN_PREFILL_GATE_OUT` saves rows.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MFERENCE_QWEN_PREFILL_GATE"] != nil))
struct QwenPrefillEquivalenceGateTests {
    struct Item {
        let name: String
        let tokens: [Int32]
        let prefillCount: Int
    }

    struct Row {
        let nll: Double
        let top1: Int
    }

    struct Outcome {
        var rows: [String: [Row]] = [:]
        var prefillSeconds: [String: Double] = [:]
        var groupedTiles = 0
        var tensorOpsAttentionLayers = 0
    }

    private static func messages(_ url: URL) throws -> [MFTokenizer.Message] {
        let rows = try JSONDecoder().decode([[String: String]].self, from: Data(contentsOf: url))
        return try rows.map {
            MFTokenizer.Message(role: try #require(MFTokenizer.Role(rawValue: $0["role"] ?? "")),
                                content: $0["content"] ?? "")
        }
    }

    @Test func fasterPrefillKeepsTeacherForcedPerplexity() async throws {
        let environment = ProcessInfo.processInfo.environment
        let install = URL(fileURLWithPath: try #require(environment["MFERENCE_QWEN_PREFILL_GATE"]))
        let prompts = GemmaPrefillGateFixture.repositoryRoot
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
        let tokenizer = try await MFTokenizer.load(forModelDirectory: install)
        func chat(_ name: String, prompt: String, answer: String, continuation: Int) throws -> Item {
            let head = try tokenizer.encodeChat(messages: try Self.messages(prompts.appendingPathComponent(prompt)))
            let tail = tokenizer.encode(try GemmaPrefillGateFixture.savedAnswer(answer), addBOS: false)
            try #require(tail.count >= continuation)
            return Item(name: name, tokens: head + Array(tail.prefix(continuation)), prefillCount: head.count)
        }
        var items: [Item] = [
            try chat("medium+qat-answer", prompt: "medium-review.json",
                     answer: "medium-review.qat", continuation: 300),
            try chat("medium+original-answer", prompt: "medium-review.json",
                     answer: "medium-review.original", continuation: 300),
            try chat("long+qat-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.qat", continuation: 200),
            try chat("long+original-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.original", continuation: 200),
        ]
        if let long = environment["MFERENCE_QWEN_PREFILL_GATE_LONG"] {
            let tokens = try tokenizer.encodeChat(messages: try Self.messages(URL(fileURLWithPath: long)))
            items.append(Item(name: "long-prompt-tail", tokens: tokens, prefillCount: tokens.count - 200))
        }
        let maxContext = 32_768
        try #require(items.allSatisfy { $0.tokens.count < maxContext })

        let context = try MetalContext()
        let model = try Model.load(directoryURL: install, device: context.device,
                                   expecting: .qwen36_35B_A3B, streamingMode: .pread(slotCount: 32))
        let vocab = model.config.vocabSize
        let logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        var values = [Float](repeating: 0, count: vocab)
        var exponent = [Float](repeating: 0, count: vocab)

        // Qwen has no final logit softcap.
        func analyze(target: Int32) -> Row {
            var source = vImage_Buffer(data: logits.contents(), height: 1,
                                       width: vImagePixelCount(vocab), rowBytes: vocab * 2)
            let count = vDSP_Length(vocab)
            var maximum: Float = 0, index: vDSP_Length = 0, sum: Float = 0
            values.withUnsafeMutableBufferPointer { v in
                var destination = vImage_Buffer(data: v.baseAddress, height: 1,
                                                width: vImagePixelCount(vocab), rowBytes: vocab * 4)
                vImageConvert_Planar16FtoPlanarF(&source, &destination, 0)
                var size = Int32(vocab)
                vDSP_maxvi(v.baseAddress!, 1, &maximum, &index, count)
                exponent.withUnsafeMutableBufferPointer { e in
                    var shift = -maximum
                    vDSP_vsadd(v.baseAddress!, 1, &shift, e.baseAddress!, 1, count)
                    vvexpf(e.baseAddress!, e.baseAddress!, &size)
                    vDSP_sve(e.baseAddress!, 1, &sum, count)
                }
            }
            return Row(nll: Double(maximum + log(sum) - values[Int(target)]), top1: Int(index))
        }

        func run(_ policy: [String: String], attention: RuntimePrefillAttentionPath) async throws -> Outcome {
            let runner = try RealForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 32,
                                                           prefillChunkTokens: 2048,
                                                           prefillAttentionPath: attention,
                                                           forceLogitsHead: true),
                gemmaPrefillPolicy: nil,
                qwenPrefillPolicy: QwenPrefillPolicy(environment: policy))
            var outcome = Outcome()
            for item in items {
                runner.reset()
                let start = Date()
                _ = try await runner.prefillChunked(tokens: item.tokens[..<item.prefillCount],
                    startPosition: 0, outputMode: .logits, config: .production(chunkTokens: 2048),
                    into: logits, onProgress: { _ in })
                outcome.prefillSeconds[item.name] = Date().timeIntervalSince(start)
                var rows = [analyze(target: item.tokens[item.prefillCount])]
                for position in item.prefillCount..<(item.tokens.count - 1) {
                    try await runner.produce(token: item.tokens[position], position: position, into: logits)
                    rows.append(analyze(target: item.tokens[position + 1]))
                }
                outcome.rows[item.name] = rows
            }
            outcome.groupedTiles = runner.prefillGroupedExpertTiles
            outcome.tensorOpsAttentionLayers = runner.prefillTensorOpsAttentionLayers
            return outcome
        }

        let rowExperts: [String: String] = ["MFERENCE_QWEN_PREFILL_ROW_EXPERTS": "1"]
        let control = try await run(rowExperts, attention: .causalTiled)
        let attentionOnly = try await run(rowExperts, attention: .fullTensorOps2DPreferred)
        let candidate = try await run([:], attention: .fullTensorOps2DPreferred)

        #expect(control.groupedTiles == 0)
        #expect(control.tensorOpsAttentionLayers == 0)
        #expect(attentionOnly.groupedTiles == 0)
        #expect(attentionOnly.tensorOpsAttentionLayers > 0, """
            full-attention prefill fell back from the tensor-ops kernel; it needs macOS 26 (MSL 4.0) \
            and a GPU where the pipeline builds
            """)
        #expect(candidate.tensorOpsAttentionLayers > 0)
        #expect(candidate.groupedTiles > 0, "the 2,940-token prompt's full 2,048-token chunk fills grouped-GEMM tiles")

        func compare(_ label: String, _ base: Outcome, _ test: Outcome) throws -> [[String: Any]] {
            var differences: [Double] = [], baseSum = 0.0, sameTop = 0
            var saved: [[String: Any]] = []
            for item in items {
                let pairs = Array(zip(try #require(base.rows[item.name]), try #require(test.rows[item.name])))
                for (a, b) in pairs {
                    differences.append(b.nll - a.nll)
                    baseSum += a.nll
                    sameTop += a.top1 == b.top1 ? 1 : 0
                }
                let itemMean = pairs.reduce(0.0) { $0 + $1.1.nll - $1.0.nll } / Double(pairs.count)
                let itemSame = pairs.reduce(0) { $0 + ($1.0.top1 == $1.1.top1 ? 1 : 0) }
                print(String(format: "[qwen-prefill-gate] %@ %@: prefill %.1f s -> %.1f s, dNLL=%+.5f, top1 %d/%d",
                             label, item.name, base.prefillSeconds[item.name] ?? 0,
                             test.prefillSeconds[item.name] ?? 0, itemMean, itemSame, pairs.count))
                saved.append(["name": item.name, "base_nll": pairs.map(\.0.nll), "test_nll": pairs.map(\.1.nll),
                              "base_top1": pairs.map(\.0.top1), "test_top1": pairs.map(\.1.top1)])
            }
            let predictions = differences.count
            let meanDifference = differences.reduce(0, +) / Double(predictions)
            // Neighbouring positions share context, so the interval uses means of
            // 10-position batches rather than treating positions as independent.
            let batches = stride(from: 0, to: predictions, by: 10).map { start -> Double in
                let batch = differences[start..<min(start + 10, predictions)]
                return batch.reduce(0, +) / Double(batch.count)
            }
            let batchMean = batches.reduce(0, +) / Double(batches.count)
            let batchVariance = batches.reduce(0) { $0 + ($1 - batchMean) * ($1 - batchMean) } / Double(batches.count - 1)
            let halfWidth = 1.96 * (batchVariance / Double(batches.count)).squareRoot()
            let agreement = Double(sameTop) / Double(predictions)
            print(String(format: "[qwen-prefill-gate] %@ predictions=%d base NLL=%.5f dNLL=%+.5f (95%% %+.5f..%+.5f) top1 agreement=%.4f",
                         label, predictions, baseSum / Double(predictions), meanDifference,
                         meanDifference - halfWidth, meanDifference + halfWidth, agreement))
            let material = meanDifference > 0.005 && meanDifference - halfWidth > 0
            #expect(!material, "\(label) dNLL=\(meanDifference) +/- \(halfWidth)")
            #expect(meanDifference <= 0.02, "\(label) dNLL=\(meanDifference)")
            #expect(agreement >= 0.98, "\(label) agreement=\(agreement)")
            return saved
        }

        print("[qwen-prefill-gate] grouped tiles=\(candidate.groupedTiles) tensor-ops attention layers=\(candidate.tensorOpsAttentionLayers)")
        let controlToAttention = try compare("control->attention", control, attentionOnly)
        let attentionToCandidate = try compare("attention->candidate", attentionOnly, candidate)
        let controlToCandidate = try compare("control->candidate", control, candidate)
        if let path = environment["MFERENCE_QWEN_PREFILL_GATE_OUT"] {
            try JSONSerialization.data(withJSONObject: ["modelID": model.modelID,
                                                        "controlToAttention": controlToAttention,
                                                        "attentionToCandidate": attentionToCandidate,
                                                        "controlToCandidate": controlToCandidate])
                .write(to: URL(fileURLWithPath: path))
        }
    }
}
