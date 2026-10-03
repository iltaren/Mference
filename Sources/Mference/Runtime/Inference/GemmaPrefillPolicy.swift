import Foundation

/// Arithmetic and batching choices for Gemma-family prefill.
///
/// The QAT checkpoint keeps MLX's FP16 reduction order (`sourceFP16`) for
/// decode, routing, normalization and sliding-window attention. Its three
/// expensive prefill matmul families (projections, shared expert, routed
/// experts) use the normal kernels: they change only floating-point summation
/// order, which measured as no perplexity change, and the source-order kernels
/// cost 3-5x at long prompts. Its full-attention prefill layers use the
/// tensor-ops kernel where it builds, which accumulates in FP32 where the
/// source rounds scores and probabilities to FP16.
struct GemmaPrefillPolicy: Equatable, Sendable {
    /// MLX FP16 reduction order outside the prefill matmuls (QAT only).
    let sourceFP16: Bool
    /// The shipped QAT profile for prefill matmuls; `MFERENCE_QAT_EXACT_PREFILL=1`.
    let prefillMatmulSourceFP16: Bool
    /// The shipped QAT profile for full-attention prefill; same switch.
    let prefillAttentionSourceFP16: Bool
    /// Batched INT4 shared expert. Source arithmetic has no batched form;
    /// `MFERENCE_GEMMA_PREFILL_LEGACY=1` also disables it.
    let batchedSharedExpert: Bool
    /// Grouped-GEMM routed experts, opt-in (`PrefillGroupedExpertGate`) and
    /// never with the legacy or exact switches.
    let groupedExperts: Bool

    init(modelID: String,
         environment: [String: String] = ProcessInfo.processInfo.environment) {
        sourceFP16 = modelID == CheckpointIdentity.gemma4QAT
        prefillMatmulSourceFP16 = sourceFP16 && environment["MFERENCE_QAT_EXACT_PREFILL"] == "1"
        prefillAttentionSourceFP16 = prefillMatmulSourceFP16
        batchedSharedExpert = !prefillMatmulSourceFP16 && environment["MFERENCE_GEMMA_PREFILL_LEGACY"] != "1"
        groupedExperts = batchedSharedExpert && PrefillGroupedExpertGate.optedIn(environment)
    }
}

/// Grouped GEMM computes whole 64-row matrix tiles per expert, so it pays only
/// while padding stays small. Measured on M2: ~17.5 us per padded row against
/// ~29.6 us per real row for the row kernel, a break-even near 1.69x padding.
/// Qwen 3.6's shape (D 2048, F 512) breaks even near 1.49x for one-block
/// experts, so the same 1.5x gate serves both.
enum PrefillGroupedExpertGate {
    static let tileRows = 64

    /// Grouped GEMM reorders floating-point sums against the per-row kernel,
    /// so Gemma 4 and Qwen 3.6 use it only with
    /// `MFERENCE_PREFILL_GROUPED_EXPERTS=1`.
    static func optedIn(_ environment: [String: String]) -> Bool {
        environment["MFERENCE_PREFILL_GROUPED_EXPERTS"] == "1"
    }

    static func usesGroupedGEMM(pairCounts: [Int]) -> Bool {
        let real = pairCounts.reduce(0, +)
        guard real > 0 else { return false }
        let padded = pairCounts.reduce(0) { $0 + ($1 + tileRows - 1) / tileRows * tileRows }
        return padded * 2 <= real * 3
    }
}
