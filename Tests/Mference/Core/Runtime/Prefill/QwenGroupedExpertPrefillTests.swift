import Foundation
import Metal
import Testing
@testable import Mference

private let qwenGroupedGEMMAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPGroupedRoutedMoE(context: context).isAvailable
}()

/// Runner-level wiring of grouped-GEMM routed experts for Qwen 3.6. The toy
/// routes every token to all eight experts, so a 64-token chunk fills whole
/// matrix tiles; both schedules must agree up to the matmul reordering.
@Suite(.serialized) struct QwenGroupedExpertPrefillTests {
    struct Harness {
        let directory: URL
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let config = ArchConfig.qwen36Toy()

        init(environment: [String: String]) throws {
            directory = try QwenToySynthetic.write()
            let context = try MetalContext()
            let model = try Model.load(directoryURL: directory, device: context.device,
                expecting: config, streamingMode: .pread(slotCount: 8))
            runner = try RealForwardRunner(model: model, context: context, maxContext: 512,
                runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                    prefillChunkTokens: 64, forceLogitsHead: true),
                gemmaPrefillPolicy: nil,
                qwenPrefillPolicy: QwenPrefillPolicy(environment: environment))
            logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                options: .storageModeShared))
        }

        func prefillRow(_ tokens: ArraySlice<Int32>) async throws -> [Float] {
            _ = try await runner.prefillChunked(tokens: tokens, startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
            return UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float16.self),
                                       count: config.vocabSize).map(Float.init)
        }
    }

    static let tokens: [Int32] = (0..<192).map { Int32(4 + ($0 * 37 + 11) % 1000) }

    @Test(.enabled(if: qwenGroupedGEMMAvailable, "Requires runtime MPP TensorOps support"))
    func groupedExpertsReproducePerRowExpertLogits() async throws {
        let rows = try Harness(environment: ["MFERENCE_QWEN_PREFILL_ROW_EXPERTS": "1"])
        let grouped = try Harness(environment: [:])
        defer {
            try? FileManager.default.removeItem(at: rows.directory)
            try? FileManager.default.removeItem(at: grouped.directory)
        }
        let expected = try await rows.prefillRow(Self.tokens[...])
        let actual = try await grouped.prefillRow(Self.tokens[...])

        #expect(rows.runner.prefillGroupedExpertTiles == 0)
        #expect(grouped.runner.prefillGroupedExpertTiles > 0)
        let finite = actual.allSatisfy { $0.isFinite }
        #expect(finite)
        let worst = zip(expected, actual).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let scale = expected.reduce(Float(0)) { max($0, abs($1)) }
        print("Qwen grouped-expert toy prefill: worst |dlogit|=\(worst), max |logit|=\(scale)")
        #expect(worst <= 0.01 * max(scale, 1), "worst=\(worst), scale=\(scale)")
        #expect(expected.indices.max { expected[$0] < expected[$1] }
            == actual.indices.max { actual[$0] < actual[$1] })
    }

    /// The activation rows of a 2,048-token chunk (8 x 512 per token) fit in
    /// the attention output (16 x 256 per token), so grouped experts add no
    /// scratch memory at Qwen 3.6's production shape.
    @Test func productionShapeBorrowsTheAttentionOutput() {
        let layout = PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 2_048,
                                               groupedExperts: true)
        #expect(layout.groupedExpertActivationElements == layout.attentionOutputElements)
        #expect(layout.dedicatedGroupedExpertActivationElements == 0)
    }
}
